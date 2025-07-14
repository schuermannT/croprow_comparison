from common_tools import calc_x
import cv2 as cv
import math
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

def gather_eval_data(log_path:str, preds:list[list[list[float]]], gts:list[list[list[float]]], filenames:list[str], fps:float, img_height:int, img_width:int):
  maize_sa_loss = []
  maize_lp_loss = []
  maize_files = []
  maize_line_ratio = []
  soy_sa_loss = []
  soy_lp_loss = []
  soy_files = []
  soy_line_ratio = []
  sugarbeet_sa_loss = []
  sugarbeet_lp_loss = []
  sugarbeet_files = []
  sugarbeet_line_ratio = []

  #calculate loss per file
  for p, g, f in tqdm(zip(preds, gts, filenames), desc='calculating losses'):
    matched = match_lines(p, g, img_height, img_width)
    sa_loss = section_angle_loss(matched, img_height, img_width)
    lp_loss = lateral_pixel_loss(matched, img_height, img_width)
    line_ratio = len(p) - len(g)
    if f[0].isnumeric(): #woodrat -> maize
      maize_sa_loss.append(sa_loss)
      maize_lp_loss.append(lp_loss)
      maize_line_ratio.append(line_ratio)
      maize_files.append(f)
    elif f[0] == 's': #crdld -> sugar beet
      sugarbeet_sa_loss.append(sa_loss)
      sugarbeet_lp_loss.append(lp_loss)
      sugarbeet_line_ratio.append(line_ratio)
      sugarbeet_files.append(f)
    else: #rosario -> soy
      soy_sa_loss.append(sa_loss)
      soy_lp_loss.append(lp_loss)
      soy_line_ratio.append(line_ratio)
      soy_files.append(f)

  #save statistical data to file
  combined_sa = maize_sa_loss + soy_sa_loss + sugarbeet_sa_loss
  combined_lp = maize_lp_loss + soy_lp_loss + sugarbeet_lp_loss
  combined_lr = maize_line_ratio + soy_line_ratio + sugarbeet_line_ratio

  with open(log_path, mode='w') as log_file:
    log_file.write(f'FPS:,{fps}\n')
    log_file.writelines(['---Section Angle Loss---\n',
                        'stat,total,maize,soy,sugarbeet\n',
                        f'mean:,{np.mean(combined_sa)},{np.mean(maize_sa_loss)},{np.mean(soy_sa_loss)},{np.mean(sugarbeet_sa_loss)}\n',
                        f'median:,{np.median(combined_sa)},{np.median(maize_sa_loss)},{np.median(soy_sa_loss)},{np.median(sugarbeet_sa_loss)}\n',
                        f'min:,{np.min(combined_sa)},{np.min(maize_sa_loss)},{np.min(soy_sa_loss)},{np.min(sugarbeet_sa_loss)}\n',
                        f'max:,{np.max(combined_sa)},{np.max(maize_sa_loss)},{np.max(soy_sa_loss)},{np.max(sugarbeet_sa_loss)}\n',
                        f'rms:,{np.sqrt(np.mean(np.square(combined_sa)))},{np.sqrt(np.mean(np.square(maize_sa_loss)))},{np.sqrt(np.mean(np.square(soy_sa_loss)))},{np.sqrt(np.mean(np.square(sugarbeet_sa_loss)))}\n'])
    log_file.writelines(['---Lateral Pixel Loss---\n',
                        'stat,total,maize,soy,sugarbeet\n',
                        f'mean:,{np.mean(combined_lp)},{np.mean(maize_lp_loss)},{np.mean(soy_lp_loss)},{np.mean(sugarbeet_lp_loss)}\n',
                        f'median:,{np.median(combined_lp)},{np.median(maize_lp_loss)},{np.median(soy_lp_loss)},{np.median(sugarbeet_lp_loss)}\n',
                        f'min:,{np.min(combined_lp)},{np.min(maize_lp_loss)},{np.min(soy_lp_loss)},{np.min(sugarbeet_lp_loss)}\n',
                        f'max:,{np.max(combined_lp)},{np.max(maize_lp_loss)},{np.max(soy_lp_loss)},{np.max(sugarbeet_lp_loss)}\n',
                        f'rms:,{np.sqrt(np.mean(np.square(combined_lp)))},{np.sqrt(np.mean(np.square(maize_lp_loss)))},{np.sqrt(np.mean(np.square(soy_lp_loss)))},{np.sqrt(np.mean(np.square(sugarbeet_lp_loss)))}\n'])
    log_file.writelines(['---Found Line Ratio---\n',
                        'stat,total,maize,soy,sugarbeet\n',
                        f'mean:,{np.mean(combined_lr)},{np.mean(maize_line_ratio)},{np.mean(soy_line_ratio)},{np.mean(sugarbeet_line_ratio)}\n',
                        f'median:,{np.median(combined_lr)},{np.median(maize_line_ratio)},{np.median(soy_line_ratio)},{np.median(sugarbeet_line_ratio)}\n',
                        f'min:,{np.min(combined_lr)},{np.min(maize_line_ratio)},{np.min(soy_line_ratio)},{np.min(sugarbeet_line_ratio)}\n',
                        f'max:,{np.max(combined_lr)},{np.max(maize_line_ratio)},{np.max(soy_line_ratio)},{np.max(sugarbeet_line_ratio)}\n'])
    log_file.write('---File Values---\n')
    log_file.write('name,section-angle-loss,lateral-pixel-loss,line-ratio\n')
    for name, sa, lp, lr in zip(maize_files, maize_sa_loss, maize_lp_loss, maize_line_ratio):
      log_file.write(f'{name},{sa},{lp},{lr}\n')
    for name, sa, lp, lr in zip(soy_files, soy_sa_loss, soy_lp_loss, soy_line_ratio):
      log_file.write(f'{name},{sa},{lp},{lr}\n')
    for name, sa, lp, lr in zip(sugarbeet_files, sugarbeet_sa_loss, sugarbeet_lp_loss, sugarbeet_line_ratio):
      log_file.write(f'{name},{sa},{lp},{lr}\n')

def section_angle_loss(matched_lines:list[tuple[list[float], list[float]]], img_height:int, img_width:int):
  sum_loss = 0.0
  real_matches = 0
  for match in matched_lines:
    if match[0] is not None \
    and match[1] is not None:
      real_matches += 1
      p_profile = get_section_angle_profile(match[0], img_height, img_width)
      g_profile = get_section_angle_profile(match[1], img_height, img_width)
      line_losses = [abs(p - g) for p, g in zip(p_profile, g_profile)]
      sum_loss += sum(line_losses) / len(line_losses)
  return (sum_loss / real_matches)

def lateral_pixel_loss(matched_lines:list[tuple[list[float], list[float]]], img_height:int, img_width:int): 
  """Delivers same info as L2-Loss, because only points with same y-value are used for the calculation"""
  sum_loss = 0.0
  real_matches = 0
  for match in matched_lines:
    if match[0] is not None \
    and match[1] is not None:
      real_matches += 1
      p_pt_line = get_point_line(match[0], img_height, img_width)
      g_pt_line = get_point_line(match[1], img_height, img_width)
      pt_losses = [abs(p[0] - g[0]) for p, g in zip(p_pt_line, g_pt_line)]
      sum_loss += sum(pt_losses) / len(pt_losses)
  return (sum_loss / real_matches)

def get_section_angle_profile(coeffs:list[float], img_height:int, img_width:int) -> list[float]:
  profile = []
  points = get_point_line(coeffs, img_height, img_width)
  for i in range(len(points)-1):
    profile.append(math.degrees(math.atan2(points[i+1][1]-points[i][1], points[i+1][0]-points[i][0])))
  return profile

def match_lines(pred:list[list[float]], gt:list[list[float]], img_height:int, img_width:int):
  pred_avgs = [avg_x(p, img_height, img_width) for p in pred]
  gt_avgs = [avg_x(g, img_height, img_width) for g in gt]
  cost_matrix = np.abs(np.subtract.outer(pred_avgs, gt_avgs))
  p_idx, g_idx = linear_sum_assignment(cost_matrix)

  out = []
  for p, g in zip(p_idx, g_idx): #add pairings to out
    out.append((pred[p], gt[g]))
  for i, p in enumerate(pred):  #add unmatched preds to out
    if i not in p_idx:
      out.append((p, None))
  for j, g in enumerate(gt):    #add unmatched gts to out
    if j not in g_idx:
      out.append((None, g))
  return out

def avg_x(coeffs:list[float], img_height:int, img_width:int):
  x_sum = 0
  for y in range(img_height):
    x = calc_x(coeffs, y)
    if 0 <= x < img_width:
      x_sum += x
  return (x_sum / img_height)

def get_point_line(coeffs:list[float], img_height:int, img_width:int, step:int=1):
  pt_line = []
  for y in range(0, img_height, step):
    x = calc_x(coeffs, y)
    if 0 <= x < img_width:
      pt_line.append((x, y))
  return pt_line

def show_lines(img:torch.Tensor, pred:torch.Tensor, gt:torch.Tensor=None):
  np_img = cv.cvtColor((img.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8), cv.COLOR_RGB2BGR)
  if gt:
    if isinstance(gt, torch.Tensor):
      gt = gt.tolist()
    for gt_line in gt:
      points = []
      for y in range(np_img.shape[0]):
        x = calc_x(gt_line, y)
        if 0 <= x < np_img.shape[1]:
          points.append((int(round(x)), y))
      if len(points) > 5:
        cv.polylines(np_img, [np.array(points)], isClosed=False, color=(0, 255, 0), thickness=2)
  if isinstance(pred, torch.Tensor):
    pred = pred.tolist()
  for pred_line in pred:
    points = []
    for y in range(np_img.shape[0]):
      x = calc_x(pred_line, y)
      if 0 <= x < np_img.shape[1]:
        points.append((int(round(x)), y))
    if len(points) > 5:
      cv.polylines(np_img, [np.array(points)], isClosed=False, color=(0, 0, 255), thickness=2)
  return cv.cvtColor(np_img, cv.COLOR_BGR2RGB)
  
from random import randint

def show_points(img, pt_lines):
  if isinstance(img, torch.Tensor):
    img = cv.cvtColor((img.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8), cv.COLOR_RGB2BGR)
  pt_color = [0, 0, 0]
  for line in pt_lines:
    pt_color[0] = randint(0, 256)
    pt_color[1] = randint(0, 256)
    pt_color[2] = randint(0, 256)
    for pt in line:
      img = cv.circle(img, (int(pt[0]), int(pt[1])), radius=2, color=pt_color, thickness=-1)
  return cv.cvtColor(img, cv.COLOR_BGR2RGB)