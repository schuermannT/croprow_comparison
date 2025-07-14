import numpy as np
import cv2 as cv
from collections import defaultdict
import torch
from typing import Any

fixed_colors = [
  (255, 0, 0),    # Blau
  (0, 255, 0),    # Grün
  (0, 0, 255),    # Rot
  (255, 255, 0),  # Cyan
  (255, 0, 255),  # Magenta
  (0, 255, 255),  # Gelb
  (128, 0, 128),  # Lila
  (0, 128, 128),  # Türkis
]

class MSECropRowFinder:
  def __init__(self, num_of_strips:int=20, poly_degree:int=5):
    self.num_of_strips = num_of_strips
    self.poly_degree = poly_degree

  def __extract_center_points(self, mask:torch.Tensor):
    height = mask.shape[0]
    strip_height = height // self.num_of_strips
    center_points_in_strip = defaultdict(list)

    for strip_idx in range(self.num_of_strips):
      y_start = strip_idx * strip_height
      if strip_idx == self.num_of_strips-1:
        y_end = height
      else:
        y_end = (strip_idx + 1) * strip_height
      strip = mask[max(0, y_start):y_end, :]

      _, _, _, centroids = cv.connectedComponentsWithStats(strip, connectivity=8)

      for c in centroids[1:]:  # skip background
        cx, cy = c
        center_points_in_strip[strip_idx].append((cx, y_start + cy))

    return center_points_in_strip

  def __classify_crop_rows(self, center_points_per_strip:defaultdict[Any, list]):
    crop_rows = []
    bottom_strip = max(center_points_per_strip.keys())

    for point in center_points_per_strip[bottom_strip]:
      crop_rows.append([point])

    for i in range(bottom_strip, min(center_points_per_strip.keys()), -1):
      current_points = center_points_per_strip[i]
      assigned = [False] * len(current_points)

      for row in crop_rows:
        last_point = row[-1]
        distances = [abs(p[0] - last_point[0]) for p in current_points]
        if distances:
          min_idx = int(np.argmin(distances))
          if not assigned[min_idx]:
            row.append(current_points[min_idx])
            assigned[min_idx] = True

      for idx, point in enumerate(current_points):
        if not assigned[idx]:
          crop_rows.append([point])

    return crop_rows

  def __fit_polynomials(self, crop_rows:list[tuple[int]]):
    polynomials = []
    for row in crop_rows:
      row = np.array(row)
      if len(row) >= self.poly_degree + 1:
        x = row[:, 0]
        y = row[:, 1]
        coeffs = np.polynomial.polynomial.polyfit(y, x, self.poly_degree)
        polynomials.append(coeffs)
    return polynomials

  def process(self, binary_tensor:torch.Tensor):
    centers = self.__extract_center_points(binary_tensor)
    crop_rows = self.__classify_crop_rows(centers)
    polynomials = self.__fit_polynomials(crop_rows)
    return polynomials, crop_rows
  
  def debug_draw_rows(self, img:cv.typing.MatLike, rows:list[list[int]]):
    copy = img.copy()
    color_counter = 0
    if len(copy.shape) == 2 or copy.shape[2] == 1:
      copy = cv.cvtColor(copy, cv.COLOR_GRAY2BGR)
    for row in rows:
      color = fixed_colors[color_counter % len(fixed_colors)]
      color_counter+=1
      if color_counter>=len(fixed_colors):
        color_counter = 0
      for pt in row:
        cv.circle(copy, (int(pt[0]), int(pt[1])), 2, color, -1)
    return copy