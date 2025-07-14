import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image, write_png
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
import random

import warnings
warnings.filterwarnings("ignore")

from common_tools import calc_x

class CropRowDataset(Dataset):
  def __init__(self, label_path:str, img_path:str, img_size:tuple[int], full_transform:bool=True):
    self.img_path = img_path
    self.label_path = label_path
    self.img_size = img_size
    self.full_transform = full_transform
    self.transform = SynchronizedTransform(img_size, full_transform)
    self.filenames = list()
    for file in os.listdir(label_path):
      if os.path.isfile(os.path.join(label_path, file)):
        self.filenames.append(file)
    

  def __len__(self):
    return len(self.filenames)
  
  def __getitem__(self, index):
    if torch.is_tensor(index):
      index = index.tolist()

    #print(self.filenames[index])
    
    img = decode_image(os.path.join(self.img_path, self.filenames[index]))
    mask = decode_image(os.path.join(self.label_path, self.filenames[index]))
    poly_label = self.load_label(os.path.join(self.label_path, os.pardir, 'labels', self.filenames[index][:-4]+'.csv'))
    poly_label = self.middle_lines(4, np.array(img).shape[1], np.array(img).shape[2], poly_label)

    if not self.full_transform:
      poly_label = self.resize_label(np.array(img).shape[-2:], self.img_size, poly_label)
    img, mask = self.transform(img, mask)
    mask = (mask > 128)

    return {
      'image': img,
      'gt': mask,
      } #'poly': poly_label} #add this line only while using the test_script
  
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

  
  def save_item(self, img, label, path):
    write_png(img, os.path.join(path, 'img.png'))
    write_png(label, os.path.join(path, 'label.png'))


class SynchronizedTransform:
  def __init__(self, img_size:tuple[int]=(480, 854), full_transform:bool=True):
    self.resize = v2.Resize(img_size)
    self.full_transform = full_transform
    

  def __call__(self, img, mask):
    # Resize (deterministisch)
    img = self.resize(img)
    mask = self.resize(mask)

    if self.full_transform:
      # RandomAffine
      scale = random.uniform(0.8, 1.2)
      img = F.affine(img, angle=0, translate=[0, 0], scale=scale, shear=[0.0, 0.0])
      mask = F.affine(mask, angle=0, translate=[0, 0], scale=scale, shear=[0.0, 0.0])

      # RandomCrop
      # i, j, h, w = v2.RandomCrop.get_params(img, output_size=(360, 640))
      # img = F.crop(img, i, j, h, w)
      # mask = F.crop(mask, i, j, h, w)

      # RandomHorizontalFlip
      if random.random() > 0.5:
        img = F.hflip(img)
        mask = F.hflip(mask)

      # RandomRotation
      angle = random.uniform(-10, 10)
      img = F.rotate(img, angle)
      mask = F.rotate(mask, angle)

      # Farbveränderungen nur für das Bild
      img = F.adjust_brightness(img, 1 + random.uniform(-0.3, 0.3))
      img = F.adjust_contrast(img, 1 + random.uniform(-0.3, 0.3))
      img = F.adjust_saturation(img, 1 + random.uniform(-0.2, 0.2))
      img = F.adjust_hue(img, random.uniform(-0.05, 0.05))

    # Typkonvertierung
    img = F.to_dtype(img, dtype=torch.float32, scale=True)
    mask = F.to_dtype(mask, dtype=torch.uint8, scale=False)

    return img, mask