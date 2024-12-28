import torch


def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.01)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class DPUnit(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.conv2 = Conv(out_ch, out_ch, k=3, p=1, g=out_ch)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class Backbone(torch.nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        # p1/2
        self.p1.append(Conv(filters[0], filters[1], k=3, s=2, p=1))
        self.p1.append(DPUnit(filters[1], filters[1]))
        # p2/4
        self.p2.append(torch.nn.MaxPool2d(kernel_size=2))
        self.p2.append(DPUnit(filters[1], filters[2]))
        self.p2.append(DPUnit(filters[2], filters[2]))
        self.p2.append(DPUnit(filters[2], filters[2]))
        # p3/8
        self.p3.append(torch.nn.MaxPool2d(kernel_size=2))
        self.p3.append(DPUnit(filters[2], filters[3]))
        self.p3.append(DPUnit(filters[3], filters[3]))
        # p4/16
        self.p4.append(torch.nn.MaxPool2d(kernel_size=2))
        self.p4.append(DPUnit(filters[3], filters[4]))
        self.p4.append(DPUnit(filters[4], filters[4]))
        # p5/32
        self.p5.append(torch.nn.MaxPool2d(kernel_size=2))
        self.p5.append(DPUnit(filters[4], filters[5]))
        self.p5.append(DPUnit(filters[5], filters[5]))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return [p3, p4, p5]


class Neck(torch.nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.up = torch.nn.Upsample(scale_factor=2)
        self.conv1 = DPUnit(filters[5], filters[4])
        self.conv2 = DPUnit(filters[4], filters[3])
        self.conv3 = DPUnit(filters[3], filters[3])

    def forward(self, x):
        p3, p4, p5 = x
        p5 = self.conv1(p5)
        p4 = self.conv2(p4 + self.up(p5))
        p3 = self.conv3(p3 + self.up(p4))
        return [p3, p4, p5]


class Head(torch.nn.Module):
    def __init__(self, filters, nc=1, nk=5):
        super().__init__()
        self.nc = nc
        self.nk = nk
        self.nl = len(filters)  # number of detection layers
        self.strides = torch.zeros(self.nl)  # strides computed during build

        self.m = torch.nn.ModuleList()
        self.cls = torch.nn.ModuleList()
        self.box = torch.nn.ModuleList()
        self.obj = torch.nn.ModuleList()
        self.kpt = torch.nn.ModuleList()

        for i in range(len(filters)):
            self.m.append(DPUnit(filters[i], filters[i]))

            self.box.append(torch.nn.Conv2d(filters[i], out_channels=4, kernel_size=1))
            self.obj.append(torch.nn.Conv2d(filters[i], out_channels=1, kernel_size=1))
            self.cls.append(torch.nn.Conv2d(filters[i], out_channels=self.nc, kernel_size=1))
            self.kpt.append(torch.nn.Conv2d(filters[i], out_channels=self.nk * 2, kernel_size=1))

    def forward(self, x):
        x = [m(i) for i, m in zip(x, self.m)]

        cls = [m(i) for i, m in zip(x, self.cls)]
        box = [m(i) for i, m in zip(x, self.box)]
        obj = [m(i) for i, m in zip(x, self.obj)]
        kpt = [m(i) for i, m in zip(x, self.kpt)]

        if self.training:
            return cls, box, obj, kpt

        n = cls[0].shape[0]
        sizes = [i.shape[2:] for i in cls]
        anchors = self.make_anchors(sizes, self.strides, cls[0].device, cls[0].dtype)

        cls = [i.permute(0, 2, 3, 1).reshape(n, -1, self.nc) for i in cls]
        box = [i.permute(0, 2, 3, 1).reshape(n, -1, 4) for i in box]
        obj = [i.permute(0, 2, 3, 1).reshape(n, -1) for i in obj]
        kpt = [i.permute(0, 2, 3, 1).reshape(n, -1, self.nk * 2) for i in kpt]

        cls = torch.cat(cls, dim=1).sigmoid()
        box = torch.cat(box, dim=1)
        obj = torch.cat(obj, dim=1).sigmoid()
        kpt = torch.cat(kpt, dim=1)

        box = self.__box_decode(torch.cat(anchors), box)
        kpt = self.__kpt_decode(torch.cat(anchors), kpt)
        return cls, box, obj, kpt

    @staticmethod
    def __box_decode(anchors, box):
        xys = (box[..., :2] * anchors[..., 2:]) + anchors[..., :2]
        whs = box[..., 2:].exp() * anchors[..., 2:]

        tl_x = (xys[..., 0] - whs[..., 0] / 2)
        tl_y = (xys[..., 1] - whs[..., 1] / 2)
        br_x = (xys[..., 0] + whs[..., 0] / 2)
        br_y = (xys[..., 1] + whs[..., 1] / 2)

        return torch.stack(tensors=[tl_x, tl_y, br_x, br_y], dim=-1)

    @staticmethod
    def __kpt_decode(anchors, kpt):
        num_kpt = int(kpt.shape[-1] / 2)
        decoded_kpt = [(kpt[..., [2 * i, 2 * i + 1]] * anchors[..., 2:]) + anchors[..., :2] for i in range(num_kpt)]

        return torch.cat(decoded_kpt, dim=-1)

    @staticmethod
    def make_anchors(sizes, strides, device, dtype, offset=0.0):
        anchors = []
        assert len(sizes) == len(strides)
        for stride, size in zip(strides, sizes):
            # keep size as Tensor instead of int, so that we can convert to ONNX correctly
            shift_x = ((torch.arange(0, size[1]) + offset) * stride).to(dtype)
            shift_y = ((torch.arange(0, size[0]) + offset) * stride).to(dtype)

            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            shift_x, shift_y = shift_x.reshape(-1), shift_y.reshape(-1)
            stride_w = shift_x.new_full((shift_x.shape[0],), stride).to(dtype)
            stride_h = shift_x.new_full((shift_y.shape[0],), stride).to(dtype)
            anchors.append(torch.stack(tensors=[shift_x, shift_y, stride_w, stride_h], dim=-1).to(device))
        return anchors


class YUNet(torch.nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.backbone = Backbone(filters)
        self.neck = Neck(filters)
        self.head = Head((filters[3], filters[3], filters[4]))

        img_dummy = torch.zeros(1, filters[0], 256, 256)
        self.head.strides = [256 / x.shape[-2] for x in self.forward(img_dummy)[0]]
        self.strides = self.head.strides

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


def version_n():
    return YUNet(filters=(3, 16, 64, 64, 64, 64))


def version_s():
    return YUNet(filters=(3, 16, 32, 64, 64, 64))


def version_t():
    return YUNet(filters=(3, 16, 32, 40, 40, 40))
