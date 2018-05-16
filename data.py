import os
import numpy as np
import torch.utils.data as torch_data
from PIL import Image as pil_image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torch
DATA_DIR = "/media/mowayao/yao_data/CarvanaImageMaskingChallenge"

class ImgMaskData(torch_data.Dataset):
    def __init__(self, names, input_size, augument=True):
        super(ImgMaskData, self).__init__()
        self.input_size = input_size
        self.augument = augument
        self.img_file_paths = map(lambda x: os.path.join(DATA_DIR, 'train/{}.jpg'.format(x)), names)
        self.mask_file_paths = map(lambda x: os.path.join(DATA_DIR, 'train_masks_png/{}_mask.png'.format(x)), names)

    def __getitem__(self, idx):
        img_path = self.img_file_paths[idx]
        mask_path = self.mask_file_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, dsize=self.input_size)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, dsize=self.input_size)
        assert np.max(mask) > 1
        mask = mask.astype(np.float32) / 255.0
        img = np.transpose(img, [2, 0, 1]) / 255.0
        if self.augument:
            img, mask = self.randomShiftScaleRotate(img, mask,  shift_limit = (-0.0625, 0.0625),
                                                    scale_limit = (0.91, 1.21), rotate_limit = (-0, 0))

        img = torch.FloatTensor(img)
        mask = torch.FloatTensor(mask)
        return img, mask

    def __len__(self):
        return len(self.img_file_paths)

    def randomHueSaturationValue(self, image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
        if np.random.random() < u:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(image)
            hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
            h = cv2.add(h, hue_shift)
            sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
            s = cv2.add(s, sat_shift)
            val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
            v = cv2.add(v, val_shift)
            image = cv2.merge((h, s, v))
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image
    def randomShiftScaleRotate(self, image, mask,
                           shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
        if np.random.random() < u:
            height, width, channel = image.shape

            angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
            scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
            aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
            sx = scale * aspect / (aspect ** 0.5)
            sy = scale / (aspect ** 0.5)
            dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
            dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

            cc = np.math.cos(angle / 180 * np.math.pi) * sx
            ss = np.math.sin(angle / 180 * np.math.pi) * sy
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])

            box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
            box1 = box0 - np.array([width / 2, height / 2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0, box1)
            image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                        borderValue=(
                                            0, 0,
                                            0,))
            mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                       borderValue=(
                                           0, 0,
                                           0,))
        return image, mask

