from __future__ import print_function, division
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
import time
import os
from core import models
import cv2
import sys
sys.path.append('.')
from AdaptiveWingLoss.utils.utils import get_preds_fromhm


class AdaptiveWingLoss:
    def __init__(self, PRETRAINED_WEIGHTS = 'AdaptiveWingLoss/ckpt/WFLW_4HG.pth', GRAY_SCALE = False, HG_BLOCKS = 4, END_RELU = False, NUM_LANDMARKS = 98):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        use_gpu = torch.cuda.is_available()
        model_ft = models.FAN(HG_BLOCKS, END_RELU, GRAY_SCALE, NUM_LANDMARKS)

        if PRETRAINED_WEIGHTS != "None":
            checkpoint = torch.load(PRETRAINED_WEIGHTS)
            if 'state_dict' not in checkpoint:
                model_ft.load_state_dict(checkpoint)
            else:
                pretrained_weights = checkpoint['state_dict']
                model_weights = model_ft.state_dict()
                pretrained_weights = {k: v for k, v in pretrained_weights.items() \
                                      if k in model_weights}
                model_weights.update(pretrained_weights)
                model_ft.load_state_dict(model_weights)

        self.model = model_ft.to(self.device)
        self.model.eval()

    def cv_crop(self, image, center, scale, resolution=256, center_shift=0):
        new_image = cv2.copyMakeBorder(image, center_shift, center_shift, center_shift, center_shift, cv2.BORDER_CONSTANT, value=[0,0,0])
        if center_shift != 0:
            center[0] += center_shift
            center[1] += center_shift
        length = 200 * scale
        center[1] += center_shift
        top = int(center[1] - length // 2)
        bottom = int(center[1] + length // 2)
        left = int(center[0] - length // 2)
        right = int(center[0] + length // 2)
        y_pad = abs(min(top, new_image.shape[0] - bottom, 0))
        x_pad = abs(min(left, new_image.shape[1] - right, 0))
        top, bottom, left, right = top + y_pad, bottom + y_pad, left + x_pad, right + x_pad
        new_image = cv2.copyMakeBorder(new_image, y_pad, y_pad, x_pad, x_pad, cv2.BORDER_CONSTANT, value=[0,0,0])
        new_image = new_image[top:bottom, left:right]
        new_image = cv2.resize(new_image, dsize=(int(resolution), int(resolution)), interpolation=cv2.INTER_LINEAR)
        return new_image

    def detect(self, image, bboxs):
        lmks = []
        with torch.no_grad():
            for box in bboxs:
                left, top, right, bottom = box[:4]
                center = [right - (right - left) / 2.0,
                        bottom - (bottom - top) / 2.0]
                center[1] = center[1] - (bottom - top) * 0.12
                scale = (right - left + bottom - top) / 195.0
                new_image = self.cv_crop(image, center, scale, 256, 0)
                if len(image.shape) == 2:
                    new_image = np.expand_dims(new_image, axis=2)
                new_image = new_image.transpose((2, 0, 1))
                new_image = np.expand_dims(new_image, axis=0)
                inputs = torch.from_numpy(new_image).float().div(255.0)
                inputs = inputs.to(self.device)
                outputs, boundary_channels = self.model(inputs)
                pred_heatmap = outputs[-1][:, :-1, :, :][0].detach().cpu()
                pred_landmarks, _ = get_preds_fromhm(pred_heatmap.unsqueeze(0), center, scale)
                pred_landmarks = pred_landmarks.squeeze().numpy()
                lmks.append(pred_landmarks)
        return lmks
