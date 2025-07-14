import numpy as np
import cv2 as cv  
from sklearn.cluster import DBSCAN

def dbscan(mask:np.array, instance_img:np.array):
  mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel=np.ones((5,5), np.uint8))
  threshold_value = 100
  masked_pixels_indices = np.where(mask > threshold_value)
  masked_pixels = instance_img[masked_pixels_indices]
  dbscan = DBSCAN()
  dbscan.fit(masked_pixels)
  labels = dbscan.labels_

  num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
  #print(f'found clusters: {num_clusters}')

  polys = []

  for cluster_idx in range(num_clusters):
    cluster_mask = (labels == cluster_idx)
    cluster_positions = (masked_pixels_indices[0][cluster_mask], masked_pixels_indices[1][cluster_mask])
    
    if len(cluster_positions[0]) >= 6:
      y = np.array(cluster_positions[0])
      x = np.array(cluster_positions[1])  
      coeffs = np.polynomial.polynomial.polyfit(y, x, 5)
      polys.append(coeffs)
  return polys

def color_based(instance_img:np.array):
  pixels = instance_img.reshape(-1, 3)
  unique_colors = np.unique(pixels, axis=0)
  pix_per_color = {}
  
  for color in unique_colors:
    mask = np.all(instance_img == color, axis=-1)
    coords = np.column_stack(np.where(mask))
    pix_per_color[tuple(color)] = coords