import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
import random

from common_tools import draw_lines_on_img, calc_x

import warnings
warnings.filterwarnings("ignore")

class MaskLessDataset(Dataset):
  def __init__(self, label_path:str, img_path:str, param_size:int, img_size:tuple[int], full_transform:bool=True):
    self.img_path = img_path
    self.label_path = label_path
    self.param_size = param_size
    self.full_transform = full_transform
    if full_transform:
      self.transform = SynchronizedTransform(img_size)
    else:
      self.transform = SynchronizedResize(img_size)
    self.filenames = list()
    for file in os.listdir(label_path):
      if os.path.isfile(os.path.join(label_path, file)):
        self.filenames.append(file)

  def __len__(self):
    return len(self.filenames)
  
  def __getitem__(self, index):
    if torch.is_tensor(index):
      index = index.tolist()
    
    img = decode_image(os.path.join(self.img_path, self.filenames[index][:-4]+'.png'))
    gt_lines = self.load_label(os.path.join(self.label_path, self.filenames[index]))
    gt_lines = self.middle_lines(self.param_size - 1, img.shape[1], img.shape[2], gt_lines)
    img, gt_lines = self.transform(img, gt_lines)
    gt_classes = [1 for _ in range(len(gt_lines))]

    if self.full_transform and gt_lines.shape[0] < self.param_size:
        gt_lines, gt_classes = self.pad_tensor(gt_lines, gt_classes, self.param_size)
    gt_classes = torch.tensor(gt_classes, dtype=torch.float32)

    #===DEBUG===
    # np_img = transformed_img.permute(1, 2, 0).detach().cpu().numpy()
    # np_img = (np_img * 255).astype(np.uint8)
    #debug_img = draw_lines_on_img(np_img, transformed_gt_lines, (0,0,255), False)
    #debug_img = draw_lines_on_img(debug_img, half_transformed, (255,0,0), False)
    #===DEBUG END===

    # if len(gt_lines) != self.param_size or len(gt_classes) != self.param_size:
    #   print(f"lines: {len(gt_lines)} | classes: {len(gt_classes)}")

    return {
      'image': img,
      'gt': gt_lines,
      'class': gt_classes
    }
  
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

  def pad_tensor(self, tensor:torch.Tensor, classes:list[int], target_B:int):
    b_diff = target_B - tensor.shape[0]
    pad = torch.tensor([[-1.0, -10000.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    for _ in range(b_diff):
      tensor = torch.cat([tensor, pad], dim=0)
      classes.append(0)
    return tensor, classes

class SynchronizedTransform:
  def __init__(self, img_size:tuple[int]):
    self.img_size = img_size
    
  def __call__(self, img:torch.Tensor, label:list[list[float]]) -> tuple[torch.Tensor, torch.Tensor]:
    ## Image Augmentation
    # Resize
    old_size = img.shape[-2:]
    resize = v2.Resize(self.img_size)
    img = resize(img)
    transformed_label = []
    half_transformed_label = []
    # Translational Shift
    translation_x = round(self.img_size[1] * random.uniform(-0.05, 0.05))
    translation_y = round(self.img_size[0] * random.uniform(-0.05, 0.05))
    img = F.affine(img, angle=0, translate=[translation_x, translation_y], scale=1.0, shear=[0.0, 0.0])
    # Horizontal Flip
    flipped = False
    if random.random() > 0.5:
      img = F.hflip(img)
      flipped = True
    # Rotation
    angle = random.uniform(-10, 10)
    img = F.rotate(img, angle)
    # Color Jittering
    img = F.adjust_brightness(img, 1 + random.uniform(-0.3, 0.3))
    img = F.adjust_contrast(img, 1 + random.uniform(-0.3, 0.3))
    img = F.adjust_saturation(img, 1 + random.uniform(-0.2, 0.2))
    img = F.adjust_hue(img, random.uniform(-0.05, 0.05))
    # Transform to type
    img = F.to_dtype(img, dtype=torch.float32, scale=True)
      
    ## Label Augmentation
    for line in label:
      # Extract Coords
      o_y = []
      o_x = []
      for y in range(old_size[0]):
        x = calc_x(line, y)
        if 0 <= x < old_size[1]:
          o_x.append(x)
          o_y.append(y)
        else:
          break
      o_y = np.array(o_y, dtype=np.float32)
      o_x = np.array(o_x, dtype=np.float32)
      # Resize
      resized_y = o_y * (self.img_size[0] / old_size[0])
      resized_x = o_x * (self.img_size[1] / old_size[1])
      # Translational Shift
      shifted_y = resized_y + translation_y
      shifted_x = resized_x + translation_x
      # Horizontal Flip
      if flipped:
        flipped_x = self.img_size[1] - 1 - shifted_x
      else:
        flipped_x = shifted_x
      flipped_y = shifted_y
      # Rotation
      cx = (self.img_size[1] - 1)/2.0
      cy = (self.img_size[0] - 1)/2.0
      angle_rad = np.deg2rad(-angle)
      x_shifted_to_centrum = flipped_x - cx
      y_shifted_to_centrum = flipped_y - cy
      x_rotated_in_centrum = x_shifted_to_centrum * np.cos(angle_rad) - y_shifted_to_centrum * np.sin(angle_rad)
      y_rotated_in_centrum = x_shifted_to_centrum * np.sin(angle_rad) + y_shifted_to_centrum * np.cos(angle_rad)
      rotated_x = x_rotated_in_centrum + cx
      rotated_y = y_rotated_in_centrum + cy
      sort_idx = np.argsort(rotated_y)
      sorted_y = rotated_y[sort_idx]
      sorted_x = rotated_x[sort_idx]
      # Fit Polynom
      if len(rotated_x) > 10:
        coeffs = np.polynomial.polynomial.polyfit(sorted_y, sorted_x, 5)
        transformed_label.append(coeffs)
    
    return img, torch.tensor(transformed_label, dtype=torch.float32)
  
class SynchronizedResize():
  def __init__(self, new_size):
    self.new_size = new_size

  def __call__(self, img:torch.Tensor, label:list[list[float]]) -> tuple[torch.Tensor, torch.Tensor]:
    old_size = img.shape[-2:]
    resize = v2.Resize(self.new_size)
    img = resize(img)
    img = F.to_dtype(img, dtype=torch.float32, scale=True)
    ## Label Augmentation
    transformed_label = []
    for line in label:
      # Extract Coords
      o_y = []
      o_x = []
      for y in range(old_size[0]):
        x = calc_x(line, y)
        if 0 <= x < old_size[1]:
          o_x.append(x)
          o_y.append(y)
        else:
          break
      o_y = np.array(o_y, dtype=np.float32)
      o_x = np.array(o_x, dtype=np.float32)
      # Resize
      resized_y = o_y * (self.new_size[0] / old_size[0])
      resized_x = o_x * (self.new_size[1] / old_size[1])
      if resized_x.size > 0:
        coeffs = np.polynomial.polynomial.polyfit(resized_y, resized_x, 5)
        transformed_label.append(coeffs)
    return img, torch.tensor(transformed_label, dtype=torch.float32)
