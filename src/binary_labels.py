import cv2 as cv
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

from common_tools import load_label, calc_x
#from sample import Line

path = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'ma_dataset', 'combined', 'val')
img_path = os.path.join(path, 'imgs')
label_path = os.path.join(path, 'labels')
out_path = os.path.join(path, 'masks')
inst_path = os.path.join(path, 'instance_masks')


def create_binary_from_labels(debug = False, instances = False):
  if not instances and not os.path.exists(out_path):
    Path(out_path).mkdir(parents=True, exist_ok=True)
  if instances and not os.path.exists(inst_path):
    Path(inst_path).mkdir(parents=True, exist_ok=True)
  with open(os.path.join(path, 'lateral_pixel_error.csv'), mode='w') as error_file:
    max_lines = 0
    label_counter = 0
    line_sum = 0
    for filename in tqdm(os.listdir(label_path)):
      if not Path(os.path.join(out_path, filename[:-4]+'.png')).is_file():
        #print(filename)
        show_img = False
        if filename == 'right_1514302336.846000.csv':
          print('debug')
          show_img = True
      file_path = os.path.join(label_path, filename)
      if os.path.isfile(file_path):
        label = load_label(file_path)
        img = cv.imread(os.path.join(img_path, filename[:-4]+'.png'))
        label = middle_lines(4, img.shape[0], img.shape[1], label)
        max_lines = max(max_lines, len(label))
        label_counter += 1
        line_sum += len(label)
        if instances:
          binary = create_instantiated_masks(label, img.shape[0], img.shape[1], thickness=3)
          save_binary_img(binary, os.path.join(path, 'instance_masks'), filename)
        else:
          binary = create_robust_masks(label, img.shape[0], img.shape[1], thickness=6)
          save_binary_img(binary, os.path.join(path, 'masks'), filename)
    print(f'average lines per image: {line_sum / label_counter}')
    print(f"Most lines per image in set: {max_lines}")

def line_intersects(bin, x, y) -> bool:
  if bin[y][x]:
    return True
  if y < bin.shape[0]-2:
    if bin[y+1][x]:
      return True
    if x > 0 and bin[y+1][x-1]:
      return True
    if x < bin.shape[1]-1 and bin[y+1][x+1]:
      return True
  return False

def save_binary_img(img, path, filename):
  if not os.path.exists(path):
    Path(path).mkdir(parents=True, exist_ok=True)
  cv.imwrite(os.path.join(path, filename[:-4]+'.png'), img)

def create_binary_img(label:list[list[float]], height:int, width:int) -> cv.typing.MatLike:
  intersect_at = -1
  bin = np.zeros(shape=(height, width, 1), dtype=np.uint8)
  for line in label:
    line_ended = False
    for y in range(height):
      x = round(calc_x(line, y))
      if x > 0 and x < width:
        if not line_intersects(bin, x, y):
          bin[y][x] = 255
        else:
          intersect_at = max(intersect_at, y)
      elif not line_ended:
        if x <= 0:
          bin[y][0] = 255
          line_ended = True
        elif x >= width:
          bin[y][width-1] = 255
          line_ended = True
  if intersect_at >= 0:
    bin[:intersect_at+1][:] = 0
  return bin

def create_robust_masks(label:list[list[float]], height:int, width:int, thickness=3) -> cv.typing.MatLike:
  mask = np.zeros((height, width), dtype=np.uint8)
  for line in label:
    points = []
    for y in range(height):
      x = calc_x(line, y)
      if 0 <= x < width:
        points.append((int(round(x)), y))
    if len(points) > 5:
      cv.polylines(mask, [np.array(points)], isClosed=False, color=255, thickness=thickness)
    else:
      print('found short line')
  cropped, _ = crop_tensor(mask, len(label))
  return cropped

def create_instantiated_masks(label:list[list[float]], height:int, width:int, thickness=3) -> cv.typing.MatLike:
  mask = np.zeros((height, width), dtype=np.uint8)
  intensities = [int(i) for i in np.linspace(10, 255, len(label), True)]
  for line, intensity in zip(label, intensities):
    points = []
    for y in range(height):
      x = calc_x(line, y)
      if 0 <= x < width:
        points.append((int(round(x)), y))
    if len(points) > 5:
      cv.polylines(mask, [np.array(points)], isClosed=False, color=intensity, thickness=thickness)
  cropped, _ = crop_tensor(mask, len(label))
  return cropped

def middle_lines(params:int, height:int, width:int, label:list[list[float]]) -> list[list[float]]:
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
    middle_lines.sort(key=lambda x: x[0])
    return middle_lines

def draw_lines_on_bin(binary: cv.typing.MatLike, lines:list[list[float]], show_img:bool=False) -> cv.typing.MatLike:
  color = cv.cvtColor(binary.copy(), cv.COLOR_GRAY2BGR)
  for line in lines:
    t_line = list()
    for y in range(binary.shape[0]):
      x = calc_x(line, y)
      if x >= 0 and x < binary.shape[1]:
        t_line.append((x, y))
    pts_arr = np.array(t_line, dtype=np.int32)
    cv.polylines(color, [pts_arr.copy()], 0, (0, 0, 255), 1)
  if show_img:
    cv.imshow('extracted lines', color)
    cv.waitKey(0)
    cv.destroyAllWindows()
  return color

def extract_poly_from_tensor(grayscale:cv.typing.MatLike, name:str) -> list[list[float]]:
  _, binary = cv.threshold(grayscale, 1, 255, cv.THRESH_BINARY)
  binary = prepare_lines(binary, name)
  if binary is None:
    return None
  contours, _ = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
  croprows = list()
  for contour in contours:
    x_coords = contour[:, 0, 0]
    y_coords = contour[:, 0, 1]
    sort_idx = np.argsort(x_coords)
    x_sorted = x_coords[sort_idx]
    y_sorted = y_coords[sort_idx]
    coeffs = np.polynomial.polynomial.polyfit(y_sorted, x_sorted, 5)
    croprows.append(coeffs)
  if not croprows:
    croprows = [[0.0] * 6]
    print('This should not be happening: Adding empty croprows')
  return croprows
    
def prepare_lines(binary:cv.typing.MatLike, name:str) -> cv.typing.MatLike:
  dilated = cv.dilate(binary, np.ones((5,5), np.uint8), iterations=4)
  binary = cv.erode(dilated, np.ones((3,3), np.uint8), iterations=3)
  binary, lines_in_img = crop_tensor(binary)
  contours, _ = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
  if len(contours) != lines_in_img:
    print(f"incorrect number of contours {len(contours)} / {lines_in_img}")
    return None
  for contour in contours:
    x_coords = contour[:, 0, 0]
    y_coords = contour[:, 0, 1]
    touches_left = (x_coords == 0).any()
    touches_right = (x_coords == binary.shape[1] - 1).any()
    touches_bottom = (y_coords == binary.shape[0] - 1).any()
    touches_top = (y_coords == 0).any()
    sort_idx = np.argsort(x_coords)
    x_sorted = x_coords[sort_idx]
    y_sorted = y_coords[sort_idx]
    coeffs = np.polynomial.polynomial.polyfit(y_sorted, x_sorted, 1)
    max_y = max(y_coords)
    min_y = min(y_coords)
    if not touches_top: #expand up
      for y in range(int(min(y_coords)), 0, -1):
        coeffs[0] = float(x_sorted[np.argmin(y_sorted)])
        x = round(calc_x(coeffs, y-min_y))
        if x < 0 or x >= binary.shape[1]:
          break
        else:
          binary[y][x] = 255
    if not (touches_left or touches_right or touches_bottom): #expand down
      for y in range(int(max(y_coords)), binary.shape[0], 1):
        coeffs[0] = float(x_sorted[np.argmax(y_sorted)])
        x = round(calc_x(coeffs, y-max_y))
        if x < 0 or x >= binary.shape[1]:
          break
        else:
          binary[y][x] = 255
  dilated = cv.dilate(binary, np.ones((5,5), np.uint8), iterations=4)
  binary = cv.erode(dilated, np.ones((3,3), np.uint8), iterations=3)
  binary, _ = crop_tensor(binary)
  return binary
  cv.imshow('binary', binary)
  cv.waitKey(0)
  cv.destroyAllWindows()


def crop_tensor(tensor_img:cv.typing.MatLike, lines_in_img:int) -> tuple[cv.typing.MatLike, int]:
  intersects_at = -1
  possible_intersect = False
  last_line_counter = 0
  for y in range(tensor_img.shape[0]):
    last_pixel = 0
    line_counter = 0
    for x in range(tensor_img.shape[1]-2):
      if last_pixel and not tensor_img[y][x]:
        line_counter = line_counter + 1
      last_pixel = tensor_img[y][x]
    if line_counter < lines_in_img:
      possible_intersect = True
      # cv.imshow('intersect', tensor_img)
      # cv.waitKey(0)
      # cv.destroyAllWindows()
    if possible_intersect\
      and line_counter > last_line_counter\
      and line_counter == lines_in_img:
      intersects_at = y
    last_line_counter = line_counter
  if intersects_at >= 0:
    tensor_img[0:intersects_at][:] = 0
      
  # for y in range(tensor_img.shape[0]):
  #   last_pixel = 0
  #   line_counter = 0
  #   for x in range(tensor_img.shape[1]):
  #     if last_pixel and not tensor_img[y][x]:
  #       line_counter = line_counter + 1
  #     last_pixel = tensor_img[y][x]
  #   if line_counter < lines_in_img:
  #     tensor_img[y][:] = 0
  #   else:
  #     break
  return tensor_img, lines_in_img

def main():
  create_binary_from_labels(debug=False, instances=True)
  #test_extract_poly_from_tensor()

if __name__ == '__main__':
  main()
