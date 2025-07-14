import os
from pathlib import Path
import numpy as np
import cv2 as cv 
from dataclasses import dataclass, field
from functools import total_ordering


separator = ', '

@total_ordering
@dataclass
class Line:
  coeffs: list[float] = field(default_factory=list)

  def get(self):
    return self.coeffs
  
  def __str__(self):
    out = ""
    for coeff in self.coeffs:
      out = out + str(coeff) + separator
    return out[:-2]
  
  def __len__(self):
    return len(self.coeffs)
  
  def __eq__(self, other):
    return self.coeffs == other.coeffs
  
  def __lt__(self, other):
    return self.coeffs[0] < other.coeffs[0]

@dataclass
class Sample:
  name: str = field(default_factory=str)
  img: cv.typing.MatLike = field(default_factory=cv.typing.MatLike)
  label: list[Line] = field(default_factory=list)

  def label_as_strings(self):
    lines = list()
    for line in self.label:
      lines.append(str(line) + '\n')
    return lines
  
  def load_label(self, path):
    label = list()
    with open(path) as label_file:
      for textline in label_file:
        coeffs = textline[:-2].strip().split(', ')
        line = Line()
        for coeff in coeffs:
          line.coeffs.append(float(coeff))
      label.append(line)
    return label
  
  def save(self, img_path:str, label_path:str):
    try:
      if not os.path.exists(img_path):
        Path(img_path).mkdir(parents=True, exist_ok=True)
      cv.imwrite(os.path.join(img_path, self.name+'.png'), self.img)
      if not os.path.exists(label_path):
        Path(label_path).mkdir(parents=True, exist_ok=True)
      with open(os.path.join(label_path, self.name+'.csv'), 'w') as label_file:
        lines = self.label_as_strings()
        label_file.writelines(lines)
    except:
      print('Could not save sample ' + self.name)