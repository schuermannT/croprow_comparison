# coding: utf-8
'''
delete the one-hot representation for instance output
'''

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import numpy as np

import random

from common_tools import calc_x


class TusimpleSet(Dataset):
  def __init__(self, dataset_path:str, img_size:tuple[int], n_labels=3, transform=True, shuffle:bool=True):
    self._gt_img_list = []
    self._gt_label_binary_list = []
    self._gt_label_instance_list = []
    self.n_labels = n_labels
    self.img_path = os.path.join(dataset_path, 'imgs')
    self.bin_path = os.path.join(dataset_path, 'masks')
    self.inst_path = os.path.join(dataset_path, 'instance_masks')
    self.poly_path = os.path.join(dataset_path, 'labels')
    self.img_size = img_size

    self.sync_transform = SynchronizedTransform(img_size, transform)

    for file in os.listdir(self.img_path):
      if os.path.isfile(os.path.join(self.img_path, file)):
        self._gt_img_list.append(file)
    for file in os.listdir(self.bin_path):
      if os.path.isfile(os.path.join(self.bin_path, file)):
        self._gt_label_binary_list.append(file)
    for file in os.listdir(self.inst_path):
      if os.path.isfile(os.path.join(self.inst_path, file)):
        self._gt_label_instance_list.append(file)
    assert len(self._gt_img_list) == len(self._gt_label_binary_list) == len(self._gt_label_instance_list)

    if shuffle:
      self._shuffle()

  def _shuffle(self):
    # randomly shuffle all list identically
    c = list(zip(self._gt_img_list, self._gt_label_binary_list, self._gt_label_instance_list))
    random.shuffle(c)
    self._gt_img_list, self._gt_label_binary_list, self._gt_label_instance_list = zip(*c)

  def __len__(self):
    return len(self._gt_img_list)

  def __getitem__(self, idx):
    # load all
    img = Image.open(os.path.join(self.img_path, self._gt_img_list[idx]))
    label_instance_img = cv2.imread(os.path.join(self.inst_path, self._gt_label_instance_list[idx]), cv2.IMREAD_UNCHANGED)
    label_img = cv2.imread(os.path.join(self.bin_path, self._gt_label_binary_list[idx]), cv2.IMREAD_COLOR)
    label_poly = self.load_label(os.path.join(self.poly_path, self._gt_img_list[idx][:-4]+'.csv'))
    label_poly = self.middle_lines(4, self.img_size[0], self.img_size[1], label_poly)

    # optional transformations
    if self._gt_img_list[idx] == 'left_1514302262.860000.png':
      pass
    label_poly = self.resize_label(np.array(img).shape, self.img_size, label_poly)
    img, label_img, label_instance_img = self.sync_transform(img, label_img, label_instance_img)

    label_binary = np.zeros([label_img.shape[0], label_img.shape[1]], dtype=np.uint8)
    mask = np.where((label_img[:, :, :] != [0, 0, 0]).all(axis=2))
    label_binary[mask] = 1

    # we could split the instance label here, each instance in one channel (basically a binary mask for each)
    return {'input':img, 
            'binary':torch.tensor(label_binary), 
            'instance':torch.tensor(label_instance_img), 
            }#'poly':label_poly} #add this line only while using the test_script
  
  def load_label(self, path:str) -> list[np.array]:
    label = list()
    with open(path) as label_file:
      for textline in label_file:
        if textline.endswith('\n'):
          textline.rstrip('\n')
        coeffs = np.array(textline.strip().split(', '), dtype=np.float32)
        label.append(coeffs)
    return label  

  def middle_lines(self, params:int, height:int, width:int, label:list[list[float]]) -> list[list[float]]:
    if len(label) <= params:
      return label
    else:
      avg_dists = []
      center_x = width // 2
      for line in label:
        # Extract Coords
        o_x = []
        for y in range(height//2, height, 1):
          x = calc_x(line, y)
          if 0 <= x < width:
            o_x.append(x)
          else:
            break
        if o_x:
          avg_dists.append((abs(int(sum(o_x) / len(o_x)) - center_x), line))
        else:
          avg_dists.append((width, line))
      avg_dists.sort(key=lambda x: x[0])
      middle_lines = [line for _, line in avg_dists[:params]]
      return middle_lines
  
  def resize_label(self, old_size:tuple[int], new_size:tuple[int], label):
    out = []
    for line in label:
      # Extract Coords
      o_y = []
      o_x = []
      for y in range(old_size[0]):
        x = calc_x(line, y)
        if 0 <= x < old_size[1]:
          o_x.append(x)
          o_y.append(y)
        elif o_x: # stop calculation if line is running out of window
          break
      o_y = np.array(o_y, dtype=np.float32)
      o_x = np.array(o_x, dtype=np.float32)
      # Resize
      resized_y = o_y * (new_size[0] / old_size[0])
      resized_x = o_x * (new_size[1] / old_size[1])
      out.append(np.polynomial.polynomial.polyfit(resized_y, resized_x, 5))

    return out

import torchvision.transforms.functional as F

class SynchronizedTransform:
  def __init__(self, img_size:tuple[int], full_transform=False):
    self.img_size = img_size
    self.full_transform = full_transform

  def __call__(self, img, bin, instance):
    if isinstance(bin, np.ndarray):
      bin = Image.fromarray(bin)
    if isinstance(instance, np.ndarray):
      instance = Image.fromarray(instance)

    img = F.resize(img, size=self.img_size)
    bin = F.resize(bin, size=self.img_size, interpolation=F.InterpolationMode.NEAREST)
    instance = F.resize(instance, size=self.img_size, interpolation=F.InterpolationMode.NEAREST)

    if self.full_transform:
      scale = random.uniform(0.8, 1.2)
      img = F.affine(img, angle=0, translate=[0, 0], scale=scale, shear=[0.0, 0.0])
      bin = F.affine(bin, angle=0, translate=[0, 0], scale=scale, shear=[0.0, 0.0])
      instance = F.affine(instance, angle=0, translate=[0, 0], scale=scale, shear=[0.0, 0.0])

      if random.random() > 0.5:
        img = F.hflip(img)
        bin = F.hflip(bin)
        instance = F.hflip(instance)

      angle = random.uniform(-10, 10)
      img = F.rotate(img, angle)
      bin = F.rotate(bin, angle)
      instance = F.rotate(instance, angle)

      img = F.adjust_brightness(img, random.uniform(0.9, 1.1))
      img = F.adjust_contrast(img, random.uniform(0.9, 1.1))
      img = F.adjust_saturation(img, random.uniform(0.9, 1.1))
      img = F.adjust_hue(img, random.uniform(-0.1, 0.1))

    img = F.to_tensor(img)
    img = F.normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    return img, np.array(bin), np.array(instance)