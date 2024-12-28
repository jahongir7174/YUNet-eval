import os
import warnings

import cv2
import numpy
import torch
import tqdm

from utils import util

warnings.filterwarnings("ignore")

data_dir = '../Dataset/WIDERFace'


def run_vga():
    size = 640
    model = torch.load(f='./weights/last.pt', map_location='cuda')
    model = model['model'].float()
    stride = int(max(model.strides))
    model.eval()

    nms = util.NMS()
    evaluator = util.Evaluator()

    results = {}
    folders = [x for x in os.listdir(f'{data_dir}/WIDER_val/images')]
    folders.sort()
    for folder in tqdm.tqdm(folders):
        if folder not in results:
            results[folder] = {}
        filenames = [x for x in os.listdir(f'{data_dir}/WIDER_val/images/{folder}')]
        filenames.sort()
        for filename in filenames:
            image = cv2.imread(f'{data_dir}/WIDER_val/images/{folder}/{filename}')
            shape = image.shape[:2]  # current shape [height, width]
            r = min(1.0, size / shape[0], size / shape[1])
            pad = int(round(shape[1] * r)), int(round(shape[0] * r))
            w = size - pad[0]
            h = size - pad[1]
            w = numpy.mod(w, stride)
            h = numpy.mod(h, stride)
            w /= 2
            h /= 2
            if shape[::-1] != pad:  # resize
                image = cv2.resize(image,
                                   dsize=pad,
                                   interpolation=cv2.INTER_LINEAR)
            top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
            left, right = int(round(w - 0.1)), int(round(w + 0.1))
            image = cv2.copyMakeBorder(image,
                                       top, bottom,
                                       left, right,
                                       cv2.BORDER_CONSTANT)  # add border
            # Convert HWC to CHW, BGR to RGB
            image = image.transpose((2, 0, 1))[::-1]
            image = numpy.ascontiguousarray(image)
            image = torch.from_numpy(image)
            image = image.unsqueeze(dim=0)
            image = image.float()
            image = image.cuda()
            # Inference
            with torch.no_grad():
                outputs = model(image)
            # NMS
            outputs = nms(outputs)
            for output in outputs:
                box_output = output[0]

                r = min(image.shape[2] / shape[0], image.shape[3] / shape[1])

                box_output[:, [0, 2]] -= (image.shape[3] - shape[1] * r) / 2  # x padding
                box_output[:, [1, 3]] -= (image.shape[2] - shape[0] * r) / 2  # y padding
                box_output[:, :4] /= r

                box_output[:, 0].clamp_(0, shape[1])  # x
                box_output[:, 1].clamp_(0, shape[0])  # y
                box_output[:, 2].clamp_(0, shape[1])  # x
                box_output[:, 3].clamp_(0, shape[0])  # y

                w = box_output[:, 2] - box_output[:, 0]
                h = box_output[:, 3] - box_output[:, 1]
                box_output[:, 2] = w
                box_output[:, 3] = h
                results[folder][filename.rstrip('.jpg')] = box_output[:, :5].cpu().numpy()
    evaluator(results)


def main():
    util.setup_seed()
    util.setup_multi_processes()

    model = torch.load(f='./weights/best.pt', map_location='cuda')
    model = model['model'].float()
    model.eval()

    nms = util.NMS()
    stride = int(max(model.strides))
    evaluator = util.Evaluator()

    results = {}
    folders = [x for x in os.listdir(f'{data_dir}/WIDER_val/images')]
    folders.sort()
    for folder in tqdm.tqdm(folders):
        if folder not in results:
            results[folder] = {}
        filenames = [x for x in os.listdir(f'{data_dir}/WIDER_val/images/{folder}')]
        filenames.sort()
        for filename in filenames:
            image = cv2.imread(f'{data_dir}/WIDER_val/images/{folder}/{filename}')
            image = image.astype(numpy.float32, copy=False)
            shape = image.shape[:2]
            pad_h = int(numpy.ceil(shape[0] / stride)) * stride
            pad_w = int(numpy.ceil(shape[1] / stride)) * stride

            top, bottom = 0, pad_h - shape[0]
            left, right = 0, pad_w - shape[1]
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
            image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            image = numpy.ascontiguousarray(image)
            image = torch.from_numpy(image)
            image = image.unsqueeze(dim=0)
            image = image.cuda()
            # Inference
            with torch.no_grad():
                outputs = model(image)
            # NMS
            outputs = nms(outputs)
            for output in outputs:
                box_output = output[0]
                box_output = box_output[:, :5].cpu().numpy()

                w = box_output[:, 2] - box_output[:, 0]
                h = box_output[:, 3] - box_output[:, 1]
                box_output[:, 2] = w
                box_output[:, 3] = h
                results[folder][filename.rstrip('.jpg')] = box_output
    evaluator(results)


if __name__ == "__main__":
    main()
