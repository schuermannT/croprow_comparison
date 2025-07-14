"""Based on the paper 'Efficient crop row detection using transformer-based parameter prediction' by Zhiming Guo"""

import torch
from torch import nn
from torchvision.models import resnet


class TransformerBasedModel(nn.Module):
  def __init__(self, d_model=256, max_crop_rows=11):
    super().__init__()
    categories = 2 #Categories contained in one image - here Background and Croprow
    poly_params = 6 #Params of the polynom - here 6 for a degree 5 polynom
    self.backbone = CNNBackBone()
    self.transformer = TransformerModule(d_model=d_model, num_queries=max_crop_rows)
    self.ffn = FeedForwardPredictionHead(d_model=d_model, hidden_dim=d_model, num_classes=categories, num_params=poly_params)

  def forward(self, x):
    features = self.backbone(x)
    transformer_out = self.transformer(features)
    class_probs, curve_params = self.ffn(transformer_out)
    return class_probs, curve_params


class CNNBackBone(nn.Module): #Based on a ResNet18 with custom channel-sizes and decreased downsampling
  def __init__(self):
    super().__init__()

    self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(16)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    self.layer1 = self._make_layer(in_channels=16, stride = 1)
    self.layer2 = self._make_layer(in_channels=16, stride = 2)
    self.layer3 = self._make_layer(in_channels=32, stride = 2)
    self.layer4 = self._make_layer(in_channels=64, stride = 2)

    
  def _make_layer(self, in_channels, stride):
    out_channels = in_channels * stride
    downsample = None
    if stride != 1:
      downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                                 nn.BatchNorm2d(out_channels))
    layers = [resnet.BasicBlock(in_channels, out_channels, stride=stride, downsample=downsample),
              resnet.BasicBlock(out_channels, out_channels)]
    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)           # → 16×180×320
    x = self.relu(self.bn1(x))   
    x = self.maxpool(x)         # → 16×90×160
    x = self.layer1(x)          # → 16×90×160
    x = self.layer2(x)          # → 32×45×80
    x = self.layer3(x)          # → 64×23×40
    x = self.layer4(x)          # → 128×12×20
    return x
  
class PositionalEncoding(nn.Module):
  def __init__(self, d_model):
    super().__init__()
    max_len = 10000 #standard value according to 'Attention is all You Need' by Vaswani et al.
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-torch.log(torch.tensor(max_len)) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)

  def forward(self, x):
    return x + self.pe[:, :x.size(0)].permute(1, 0, 2)
  
class TransformerModule(nn.Module):
  def __init__(self, d_model=256, num_queries=100):
    super().__init__()
    self.d_model = d_model
    self.linear_proj = nn.Linear(128, d_model)  # Project ResNet feature channels (128) to d_model

    self.transformer = nn.Transformer(
      d_model=d_model,
      nhead=8, #must be an integer divider of d_model
      num_encoder_layers=2,
      num_decoder_layers=2,
      dim_feedforward=512
    )

    self.query_embed = nn.Embedding(num_queries, d_model)
    self.pos_encoder = PositionalEncoding(d_model)

  def forward(self, feature_map):
    B, C, H, W = feature_map.shape
    src = feature_map.flatten(2).permute(2, 0, 1)
    src = self.linear_proj(src)
    src = self.pos_encoder(src)

    tgt = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  

    output = self.transformer(src, tgt)  
    return output.permute(1, 0, 2)  

class FeedForwardPredictionHead(nn.Module):
    def __init__(self, d_model=256, hidden_dim=256, num_classes=2, num_params=6):
        super().__init__()
        
        # Classification branch
        self.class_branch = nn.Linear(d_model, num_classes)

        # Curve param branch
        self.curve_branch = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_params)
        )

    def forward(self, decoder_output):
        class_logits = self.class_branch(decoder_output)
        class_probs = nn.functional.softmax(class_logits, dim=-1)
        
        curve_params = self.curve_branch(decoder_output)

        return class_probs, curve_params





  