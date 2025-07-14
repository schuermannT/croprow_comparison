import os
import cv2 as cv
from cv2.typing import MatLike
#from shapely import Point
import numpy as np

#from sample import Line

def load_images_from_directory(dirpath:str) -> tuple[list[MatLike], list[str]]:
    imgs = list[MatLike]()
    filenames = list[str]()
    for filename in os.listdir(dirpath):
      _img = cv.imread(os.path.join(dirpath, filename))
      if _img is not None:
        imgs.append(_img)
        filenames.append(filename)
    print('finished loading images')
    return imgs, filenames

def load_label(path):
  label = list()
  with open(path) as label_file:
    for textline in label_file:
      coeffs = textline.strip(' \n').split(', ')
      coeffs = list(map(float, coeffs))
      label.append(coeffs)
  label.sort()
  return label

def calc_x(line, y):
  x = 0
  for exp in range(len(line)):
    x = x + line[exp]*pow(y, exp)
  return float(x)

def draw_lines_on_img(img: cv.typing.MatLike, lines:list[list[float]], color:tuple[int], show_img:bool=False) -> cv.typing.MatLike:
  bgr_img = cv.cvtColor(img.copy(), cv.COLOR_RGB2BGR)
  for line in lines:
    t_line = list()
    workable_line = line.tolist()
    for y in range(img.shape[0]):
      x = calc_x(workable_line, y)
      if 0 <= x < img.shape[1]:
        t_line.append((x, y))
    pts_arr = np.array(t_line, dtype=np.int32)
    cv.polylines(bgr_img, [pts_arr.copy()], 0, color, 1)
    if show_img:
      cv.imshow('extracted lines', bgr_img)
      cv.waitKey(0)
      cv.destroyAllWindows()
  return cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)

#---------------Tools---------------

def apply_roi(img:MatLike, roi):
   #TODO
   pass

def calc_excess_green_index(img:MatLike) -> MatLike:
  """calculates the exG-Index of an Image

      params:
        img: Image in RGB-Colorspace"""
  _img = img.astype('int32')
  #extraxt color channels from image
  r = _img[:, :, 0]
  cv.imshow('r', r)
  g = _img[:, :, 1]
  cv.imshow('g', g)
  b = _img[:, :, 2]
  cv.imshow('b', b)
  #calculate excess green index
  exG_idx = 2*g - r - b  
  exG_idx[exG_idx < 0] = 0
  exG_idx = exG_idx.astype('uint8')
  print('calculated ExG')
  return exG_idx

def apply_otsu(img:MatLike) -> tuple[float, MatLike]:
  """calculates OTSU Threshold to the given Image and returns a binarized version of the image

      params:
        img: a grayscale Image. Preferably exG"""
  blur = cv.GaussianBlur(img, (5, 5), 0)
  return cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
  
# def lsm(points:list[Point], degree) -> Line:
#   x_coords = list()
#   y_coords = list()
#   for point in points:
#     x_coords.append(point.x)
#     y_coords.append(point.y)
#   coeffs = np.polynomial.polynomial.polyfit(y_coords, x_coords, degree)
#   line = Line()
#   for c in coeffs:
#     line.coeffs.append(c)
#   return line

# def add_prediction_point(row:list[Point], img_height:int):
#   simple_line = lsm(row, 1)
#   for y in range(int(row[-1].y), img_height+1, 10):
#     x = calc_x(simple_line.get(), y)
#     row.append(Point(x, y))

def lines_in_img(filepath:str):
  with open(filepath) as file:
    content = file.read()
    return content.count('\n')