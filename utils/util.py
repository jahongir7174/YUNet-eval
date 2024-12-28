import random
from multiprocessing import Pool

import numpy
import scipy
import torch
import torchvision


def setup_seed():
    """
    Setup random seed.
    """
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_multi_processes():
    """
    Setup multi-processing environment variables.
    """
    import cv2
    from os import environ
    from platform import system

    # set multiprocess start method as `fork` to speed up the training
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('fork', force=True)

    # disable opencv multithreading to avoid system being overloaded
    cv2.setNumThreads(0)

    # setup OMP threads
    if 'OMP_NUM_THREADS' not in environ:
        environ['OMP_NUM_THREADS'] = '1'

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in environ:
        environ['MKL_NUM_THREADS'] = '1'


class Evaluator:
    def __init__(self):
        self.iou_threshold = 0.5

    def __call__(self, output):
        output = self.norm_score(output)
        thresh_num = 1000
        facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = self.load_gt()
        event_num = len(event_list)
        settings = ['Easy', 'Medium', 'Hard']
        setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]

        aps = [-1.0, -1.0, -1.0]
        pool = Pool(8)

        for setting_id in range(3):
            gt_list = setting_gts[setting_id]
            count_face = 0
            pr_curve = numpy.zeros((thresh_num, 2)).astype('float')

            for i in range(event_num):
                event_name = str(event_list[i][0][0])
                img_list = file_list[i][0]
                output_list = output[event_name]
                sub_gt_list = gt_list[i][0]
                gt_bbx_list = facebox_list[i][0]

                for j in range(len(img_list)):
                    img_name = str(img_list[j][0][0])
                    output_info = output_list[img_name]

                    gt_boxes = gt_bbx_list[j][0].astype('float')
                    keep_index = sub_gt_list[j][0]
                    count_face += len(keep_index)

                    if len(gt_boxes) == 0 or len(output_info) == 0:
                        continue

                    ignore = numpy.zeros(gt_boxes.shape[0], dtype=int)
                    if len(keep_index) != 0:
                        ignore[keep_index - 1] = 1
                    recall, proposal_list = self.compute_pr(output_info,
                                                            gt_boxes,
                                                            ignore,
                                                            self.iou_threshold, pool)

                    pr_curve_single = numpy.zeros((thresh_num, 2)).astype('float')
                    for t in range(thresh_num):

                        thresh = 1 - (t + 1) / thresh_num
                        r_index = numpy.where(output_info[:, 4] >= thresh)[0]
                        if len(r_index) == 0:
                            pr_curve_single[t, 0] = 0
                            pr_curve_single[t, 1] = 0
                        else:
                            r_index = r_index[-1]
                            p_index = numpy.where(proposal_list[:r_index + 1] == 1)[0]
                            pr_curve_single[t, 0] = len(p_index)  # valid output number
                            pr_curve_single[t, 1] = recall[r_index]  # valid gt number

                    pr_curve += pr_curve_single
            pr_curve = self.compute_pr_curve(thresh_num, pr_curve, count_face)
            propose = pr_curve[:, 0]
            recall = pr_curve[:, 1]

            aps[setting_id] = self.compute_ap(recall, propose)
            print('%s AP: %.2f' % (settings[setting_id], aps[setting_id]))

        return aps

    def compute_pr(self, output, gt, ignore, iou_thresh, mpp):
        _output = output.copy()
        _gt = gt.copy()
        recall_list = numpy.zeros(_gt.shape[0])
        proposal_list = numpy.ones(_output.shape[0])
        output_recall = numpy.zeros(_output.shape[0])

        _output[:, 2] = _output[:, 2] + _output[:, 0]
        _output[:, 3] = _output[:, 3] + _output[:, 1]
        _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
        _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

        gt_overlap_list = mpp.starmap(self.compute_iou,
                                      zip([_gt] * _output.shape[0], [_output[h] for h in range(_output.shape[0])]))

        for h in range(_output.shape[0]):

            gt_overlap = gt_overlap_list[h]
            max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()

            if max_overlap >= iou_thresh:
                if ignore[max_idx] == 0:
                    recall_list[max_idx] = -1
                    proposal_list[h] = -1
                elif recall_list[max_idx] == 0:
                    recall_list[max_idx] = 1

            r_keep_index = numpy.where(recall_list == 1)[0]
            output_recall[h] = len(r_keep_index)

        return output_recall, proposal_list

    @staticmethod
    def compute_ap(rec, pre):
        # correct AP calculation
        # first append sentinel values at the end
        m_rec = numpy.concatenate(([0.], rec, [1.]))
        m_pre = numpy.concatenate(([0.], pre, [0.]))

        # compute the precision envelope
        for i in range(m_pre.size - 1, 0, -1):
            m_pre[i - 1] = numpy.maximum(m_pre[i - 1], m_pre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = numpy.where(m_rec[1:] != m_rec[:-1])[0]

        # and sum (\Delta recall) * pre
        return numpy.sum((m_rec[i + 1] - m_rec[i]) * m_pre[i + 1]) * 100

    @staticmethod
    def load_gt():
        gt_mat = scipy.io.loadmat('./data/wider_face.mat')
        hard_mat = scipy.io.loadmat('./data/wider_hard.mat')
        medium_mat = scipy.io.loadmat('./data/wider_medium.mat')
        easy_mat = scipy.io.loadmat('./data/wider_easy.mat')

        facebox_list = gt_mat['face_bbx_list']
        event_list = gt_mat['event_list']
        file_list = gt_mat['file_list']

        hard_gt_list = hard_mat['gt_list']
        medium_gt_list = medium_mat['gt_list']
        easy_gt_list = easy_mat['gt_list']

        return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list

    @staticmethod
    def norm_score(output):
        max_score = -1
        min_score = 2
        for _, k in output.items():
            for _, v in k.items():
                if len(v) == 0:
                    continue
                _min = numpy.min(v[:, -1])
                _max = numpy.max(v[:, -1])
                max_score = max(_max, max_score)
                min_score = min(_min, min_score)

        diff = max_score - min_score
        for _, k in output.items():
            for _, v in k.items():
                if len(v) == 0:
                    continue
                v[:, -1] = (v[:, -1] - min_score).astype(numpy.float64) / diff
        return output

    @staticmethod
    def compute_iou(a, b):
        x1 = numpy.maximum(a[:, 0], b[0])
        y1 = numpy.maximum(a[:, 1], b[1])
        x2 = numpy.minimum(a[:, 2], b[2])
        y2 = numpy.minimum(a[:, 3], b[3])
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        inter = w * h
        a_area = (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1)
        b_area = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
        o = inter / (a_area + b_area - inter)
        o[w <= 0] = 0
        o[h <= 0] = 0
        return o

    @staticmethod
    def compute_pr_curve(thresh_num, pr_curve, count_face):
        _pr_curve = numpy.zeros((thresh_num, 2))
        for i in range(thresh_num):
            _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
            _pr_curve[i, 1] = pr_curve[i, 1] / count_face
        return _pr_curve


class NMS:

    def __call__(self, outputs):
        cls, box, obj, kpt = outputs
        assert len(cls) == len(box) == len(obj) == len(kpt)

        outputs = []
        for i in range(cls.shape[0]):
            outputs.append(self.__nms(cls[i],
                                      box[i],
                                      obj[i],
                                      kpt[i]))
        return outputs

    def __nms(self, cls, box, obj, kpt):
        scores, indices = torch.max(cls, 1)
        valid_mask = obj * scores >= 0.001

        box = box[valid_mask]
        kpt = kpt[valid_mask]
        scores = scores[valid_mask] * obj[valid_mask]
        indices = indices[valid_mask]

        if indices.numel() == 0:
            return box, indices, kpt
        else:
            box, keep = self.__batched_nms(box, scores, indices)
            return torch.cat(tensors=(box, indices[keep][:, None]), dim=-1), kpt[keep]

    @staticmethod
    def __batched_nms(boxes, scores, indices):
        max_coordinate = boxes.max()
        offsets = indices.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]

        if len(boxes_for_nms) < 10_000:
            keep = torchvision.ops.nms(boxes_for_nms, scores, iou_threshold=0.45)
            boxes = boxes[keep]
            scores = scores[keep]
        else:
            total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
            for i in torch.unique(indices):
                mask = (indices == i).nonzero(as_tuple=False).view(-1)
                keep = torchvision.ops.nms(boxes_for_nms[mask], scores[mask], iou_threshold=0.45)
                total_mask[mask[keep]] = True

            keep = total_mask.nonzero(as_tuple=False).view(-1)
            keep = keep[scores[keep].argsort(descending=True)]
            boxes = boxes[keep]
            scores = scores[keep]

        return torch.cat(tensors=[boxes, scores[:, None]], dim=-1), keep
