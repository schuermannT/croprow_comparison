import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from common_tools import calc_x


class HungarianLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.w1 = 2
    self.w2 = 4
    self.w3 = 1

  def forward(self, pred_params_batch:torch.Tensor, pred_classes_batch:torch.Tensor, gt_params_batch:torch.Tensor, gt_classes_batch:torch.Tensor, img_height:int):
    B = pred_params_batch.shape[0]
    total_loss = 0.0

    cost_matrices = self.compute_cost_matrix(pred_classes_batch, pred_params_batch, gt_classes_batch, gt_params_batch, img_height)

    for i in range(B):
      cost_matrix = cost_matrices[i].cpu().detach().numpy()

      gt_idx, pred_idx = linear_sum_assignment(cost_matrix)

      loss = 0.0
      for g, p in zip(gt_idx, pred_idx):
        gt_class = int(gt_classes_batch[i, g].item())
        class_loss = pred_classes_batch[i, p, gt_class]
        loss += -self.w1 * class_loss
        if gt_class == 1:
          pred_pts = self.polys_to_pts_tensor(pred_params_batch[i][p:p+1], img_height)[0]
          gt_pts = self.polys_to_pts_tensor(gt_params_batch[i][g:g+1], img_height)[0]
          loss += self.w2 * nn.functional.l1_loss(pred_pts, gt_pts)
          loss += self.w3 * nn.functional.l1_loss(pred_params_batch[i, p], gt_params_batch[i, g])
      total_loss += loss
    return total_loss / B

  def compute_cost_matrix(self, pred_classes:torch.Tensor, pred_params:torch.Tensor, gt_classes:torch.Tensor, gt_params:torch.Tensor, img_height:int):
    B, N, _ = pred_params.shape
    cost_matrix = torch.zeros(B, N, N, device=pred_params.device)

    for i in range(B):
      pred_pts = self.polys_to_pts_tensor(pred_params[i], img_height)
      gt_pts = self.polys_to_pts_tensor(gt_params[i], img_height)

      for gt_idx in range(N):
        for pred_idx in range(N):
          is_crop = gt_classes[i, gt_idx]
          class_cost = -pred_classes[i, pred_idx, int(is_crop.item())]
          path_cost = torch.nn.functional.l1_loss(gt_pts[gt_idx], pred_pts[pred_idx], reduction='sum')
          param_cost = torch.nn.functional.l1_loss(gt_params[i, gt_idx], pred_params[i, pred_idx], reduction='sum')
          cost_matrix[i, gt_idx, pred_idx] = (self.w1 * class_cost +
                                              is_crop * self.w2 * path_cost +
                                              is_crop * self.w3 * param_cost)
    return cost_matrix  
  
  def polys_to_pts_tensor(self, params: torch.Tensor, img_height:int) -> torch.Tensor:
    lines = []
    for line_params in params:
      y_vals = torch.arange(0, img_height, 10, device=params.device).float()
      x_vals = self.calc_x_tensor(line_params, y_vals)
      pts = torch.stack([x_vals, y_vals], dim=1)
      lines.append(pts)
    return torch.stack(lines)
  
  def calc_x_tensor(self, coeffs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    powers = torch.stack([y ** i for i in range(coeffs.shape[0])], dim=0)
    x = torch.matmul(coeffs, powers)
    return x


  def pad_tensor(self, params:torch.Tensor, classes:torch.Tensor, target_B:int):
    b_diff = target_B - params.shape[0]
    pad = torch.tensor([[-1.0, -10000.0, 0.0, 0.0, 0.0, 0.0]], dtype=params.dtype, device=params.device)
    c_pad = torch.tensor([0], dtype=classes.dtype, device=classes.device)
    for _ in range(b_diff):
      params = torch.cat([params, pad], dim=0)
      classes = torch.cat([classes, c_pad], dim=0)
    return params, classes

  def match_lines(self, pred_lines, gt_lines):
    B, N, _ = pred_lines.shape
    diffs = torch.abs(pred_lines[:, None] - gt_lines[None, :])
    cost_matrix = diffs.sum(dim=(-1, -2))
    row_idx, col_idx = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
    return torch.tensor(row_idx, dtype=torch.long), torch.tensor(col_idx, dtype=torch.long)
  
  def polys_to_pts(self, params:list[list[float]], img_height:int) -> torch.Tensor:
    lines = list()
    for l in params:
      line = list()
      for y in range(0, img_height, 10):
        line.append((calc_x(l, y), y))
      lines.append(line)
    return torch.tensor(lines, dtype=torch.float32)

        
        
# https://gist.github.com/ivanstepanovftw/e079280976f2466e58f3200e14d9e3a1