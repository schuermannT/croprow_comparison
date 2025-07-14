import os
from pathlib import Path
import cv2 as cv
import numpy as np
from shapely import Point

from sample import Sample, Line

class LabelMaker():
  def __init__(self, img_path:str, pre_mask_path:str):
    self.img_path = img_path
    self.pre_mask_path = pre_mask_path
    self.filenames = list()
    self.flag_save_file = True
    for file in os.listdir(pre_mask_path):
      if os.path.isfile(os.path.join(pre_mask_path, file)):
        self.filenames.append(file)

  def get_img(self, filename:str) -> cv.typing.MatLike:
    return cv.imread(os.path.join(self.img_path, filename+'.png'))
  
  def get_pre_mask(self, filename:str) -> list[list[Point]]:
    pre_mask_path = os.path.join(self.pre_mask_path, filename+'.csv')
    rows = list()
    with open(pre_mask_path, mode='r') as file:
      croprow = list()
      last_row_added = False
      for textline in file:
        last_row_added = False
        point = Point(textline.strip().split(', '))
        if point.x != -1:
          croprow.append(point)
        else:
          if len(croprow) > 0:
            rows.append(croprow)
            croprow = list()
            last_row_added = True
      if not last_row_added and len(croprow) > 0:
        rows.append(croprow)
    return rows
  
  def lsm(self, points:list[Point], degree) -> Line:
    x_coords = list()
    y_coords = list()
    for point in points:
      x_coords.append(point.x)
      y_coords.append(point.y)
    coeffs = np.polynomial.polynomial.polyfit(y_coords, x_coords, degree)
    line = Line()
    for c in coeffs:
      line.coeffs.append(c)
    return line
  
  def add_prediction_point(self, row:list[Point], img_height:int):
    simple_line = self.lsm(row, 1)
    for y in range(int(row[-1].y), img_height+1, 10):
      x = calc_x(simple_line.get(), y)
      row.append(Point(x, y))

  def click_event(self, event, x, y, flags, params):
    if event != 0 and event == cv.EVENT_LBUTTONDOWN:
      self.flag_save_file = False
      print('image will not be saved')
    elif event != 0 and event == cv.EVENT_MBUTTONDOWN:
      self.flag_save_file = True
      print('image will be saved')
  
def calc_x(line, y):
  x = 0
  for exp in range(len(line)):
    x = x + line[exp]*pow(y, exp)
  return x
  
def main():
  base_path = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'ma_dataset', 'crdld_test')
  img_path = os.path.join(base_path, 'imgs')
  pre_mask_path = os.path.join(base_path, 'pre_masks')
  save_label_path = os.path.join(os.path.dirname(__file__), os.pardir, 'labels')
  save_line_img_path = os.path.join(os.path.dirname(__file__), os.pardir, 'line_imgs')
  lm = LabelMaker(img_path, pre_mask_path)
  if not os.path.exists(save_label_path):
    Path(save_label_path).mkdir(parents=True, exist_ok=True)
  for filename in lm.filenames:
    filename = filename[:-4]
    if not Path(os.path.join(save_label_path, filename+'.csv')).is_file():
      rows = lm.get_pre_mask(filename)
      img = lm.get_img(filename)
      sample = Sample(filename, img)
      print(len(rows))
      for row in rows:
        lm.add_prediction_point(row, img.shape[0])
        line = lm.lsm(row, 5)
        sample.label.append(line)
        
        pts = list()
        for y in range(img.shape[0]):
          x = calc_x(line.get(), y)
          pts.append((x, y))
        pts_arr = np.array(pts, dtype=np.int32)
        cv.polylines(img, [pts_arr], 0, (0, 0, 255), 3)
      lm.flag_save_file = True
      cv.imshow(filename, img)
      cv.setMouseCallback(filename, lm.click_event)
      cv.waitKey(0)
      cv.destroyAllWindows()
      if lm.flag_save_file:
        sample.img = img
        sample.save(save_line_img_path, save_label_path)
        print('labeled and saved ' + sample.name)
    else:
      print(f"skipped {filename}")

if __name__ == '__main__':
  main()
