"""based on the paper 'Recognition method of maize crop rows at the seedling stage based on MS-ERFNet model' by Ciangnan Liu"""

import torch
import torch.nn as nn


class MSERFNet(nn.Module):
  def __init__(self):
    super().__init__()
    #Encoder
    self.down1 = DownsamplingBlock(3, 16) #out-shape: 256x256x16
    self.ms1 = MSModule(16, groups=4) # groups = channels / 4
    self.down2 = DownsamplingBlock(16, 32) #out-shape: 128x128x32
    self.ms2 = MSModule(32)
    self.down3 = DownsamplingBlock(32, 64) #out-shape: 64x64x64
    self.resa = RESA(64)
    self.ms3 = MSModule(64)
    self.down4 = DownsamplingBlock(64, 128) #out-shape: 32x32x128
    self.ms4 = MSModule(128)
    self.down5 = DownsamplingBlock(128, 256) #out-shape: 16x16x256
    self.ms5 = MSModule(256)
    self.down6 = DownsamplingBlock(256, 512) #out-shape: 8x8x512
    #Decoder
    self.up5 = UpsamplingBlock(512, 256) #out-shape: 16x16x256
    self.fuse5 = FusionBlock(256, 512)   #out-shape: 16x16x512
    self.up4 = UpsamplingBlock(512, 128) #out-shape: 32x32x128
    self.fuse4 = FusionBlock(128, 256)   #out-shape: 32x32x256
    self.up3 = UpsamplingBlock(256, 64)  #out-shape: 64x64x64
    self.fuse3 = FusionBlock(64, 128)    #out-shape: 64x64x128
    self.up2 = UpsamplingBlock(128, 32)  #out-shape: 128x128x32
    self.fuse2 = FusionBlock(32, 64)     #out-shape: 128x128x64
    self.up1 = UpsamplingBlock(64, 16)   #out-shape: 256x256x16
    self.fuse1 = FusionBlock(16, 32)     #out-shape: 256x256x32 !!! This step is not shown in the paper, but got included to match the previous pattern and maximise detail preservation
    self.up0 = UpsamplingBlock(32, 2)    #out-shape: 512x512x2  !!! This step is not shown in the paper, but got included to reach the input format of 512x512 and binary channel

  def forward(self, x) -> torch.Tensor:
    #Encoder
    x_d1 = self.down1(x)
    x_ms1 = self.ms1(x_d1)
    x_d2 = self.down2(x_ms1)
    x_ms2 = self.ms2(x_d2)
    x_d3 = self.down3(x_ms2)
    x_resa = self.resa(x_d3)
    x_ms3 = self.ms3(x_resa)
    x_d4 = self.down4(x_ms3)
    x_ms4 = self.ms4(x_d4)
    x_d5 = self.down5(x_ms4)
    x_ms5 = self.ms5(x_d5)
    x_d6 = self.down6(x_ms5)
    #Decoder
    x_u5 = self.up5(x_d6)
    x_f5 = self.fuse5(x_u5, x_d5)
    x_u4 = self.up4(x_f5)
    x_f4 = self.fuse4(x_u4, x_d4)
    x_u3 = self.up3(x_f4)
    x_f3 = self.fuse3(x_u3, x_d3)
    x_u2 = self.up2(x_f3)
    x_f2 = self.fuse2(x_u2, x_d2)
    x_u1 = self.up1(x_f2)
    x_f1 = self.fuse1(x_u1, x_d1) # This and the following upsampling are not shown in the papers fig. 5. They were added due to common sense
    x_u0 = self.up0(x_f1)
    return x_u0

class FusionBlock(nn.Module):
  def __init__(self, in_channels:int, out_channels:int):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(True)

  def forward(self, x1, x2):
    x = x1 + x2
    return self.relu(self.bn(self.conv(x)))

class DownsamplingBlock(nn.Module):
  def __init__(self, in_channels:int, out_channels:int):
    super().__init__()
    # Convolution Branch
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
    self.bn_conv = nn.BatchNorm2d(out_channels)
    # Maxpooling Branch
    self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.pool_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    self.bn_pool = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU()

  def forward(self, x):
    cx = self.conv(x)
    conv_out = self.bn_conv(cx)
    px = self.pool_projection(self.max_pool(x))
    pool_out = self.bn_pool(px)
    return self.relu(conv_out + pool_out)

class UpsamplingBlock(nn.Module):
  def __init__(self, in_channels:int, out_channels:int):
    super().__init__()
    self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(True)

  def forward(self, x):
    x = self.relu(self.bn(self.deconv(x)))
    return x

class MSModule(nn.Module):
  def __init__(self, channels:int, dilation:int=None, groups:int=None):
    super().__init__()

    if dilation is None:
      dilation = 2 if channels >= 64 else 1
    if groups is None:
      groups = max(1, channels // 8)
    
    # Parallel Structure
    self.para_conv_1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1)
    self.para_conv_3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
    self.para_conv_5 = nn.Sequential(
      nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2, groups=channels), #depthwise
      nn.Conv2d(channels, channels, kernel_size=1, stride=1) #pointwise
    )
    self.parallel_reduce = nn.Conv2d(3*channels, channels, kernel_size=1, stride=1)

    # Serial Structure
    self.serial_asymm1_1 = nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=1, padding=(0, 1), groups=groups)
    self.serial_asymm1_2 = nn.Conv2d(channels, channels, kernel_size=(3, 1), stride=1, padding=(1, 0), groups=groups)
    self.serial_asymm2_1 = nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=1, padding=(0, dilation), dilation=(1, dilation), groups=groups)
    self.serial_asymm2_2 = nn.Conv2d(channels, channels, kernel_size=(3, 1), stride=1, padding=(dilation, 0), dilation=(dilation, 1), groups=groups)
    self.serial_reduce = nn.Conv2d(channels, channels, kernel_size=1, stride=1)

  def forward(self, x):
    # Parallel Structure
    para_out1 = self.para_conv_1(x)
    para_out2 = self.para_conv_3(x)
    para_out3 = self.para_conv_5(x)
    para_concat = torch.cat([para_out1, para_out2, para_out3], dim=1)
    parallel_out = self.parallel_reduce(para_concat)
    # Serial Structure
    serial_interim1 = self.serial_asymm1_1(parallel_out)
    serial_out1 = self.serial_asymm1_2(serial_interim1)
    serial_interim2 = self.serial_asymm2_1(serial_out1)
    serial_out2 = self.serial_asymm2_2(serial_interim2)
    serial_out = self.serial_reduce(serial_out2)
    # Residual Addition
    return x + serial_out

class RESA(nn.Module):
  def __init__(self, channels:int=64):
    super().__init__()
    self.kernel_size = 9 #according to original RESA: http://refhub.elsevier.com/S0168-1699(23)00352-6/h0255
    self.padding = self.kernel_size // 2
    self.K = int(torch.log2(torch.tensor(64))) # = 6 with H=64 (height)

    # Vertical 1D-Convs per Iteration
    self.conv_v = nn.ModuleList([nn.Conv2d(channels, 
                                           channels, 
                                           kernel_size=(self.kernel_size, 1), 
                                           padding=(self.padding, 0),
                                           groups=channels,
                                           bias=False)
                                  for _ in range(self.K)
                                ])
    # Horizontal 1D-Convs per Iteration
    self.conv_h = nn.ModuleList([nn.Conv2d(channels,
                                           channels,
                                           kernel_size=(1, self.kernel_size),
                                           padding=(0, self.padding),
                                           groups=channels,
                                           bias=False)
                                  for _ in range(self.K)
                                ])
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    out = x
    H = x.shape[2]
    for k in range(self.K):
      s_k = H // (2 ** (self.K - 1 - k)) 

      # Vertical Shifting
      v_down = torch.roll(out, shifts=s_k, dims=2)
      v_up = torch.roll(out, shifts=-s_k, dims=2)
      v_feat = self.conv_v[k](v_down) + self.conv_v[k](v_up)

      # Horizontal Shifting
      h_right = torch.roll(out, shifts=s_k, dims=3)
      h_left = torch.roll(out, shifts=-s_k, dims=3)
      h_feat = self.conv_h[k](h_right) + self.conv_h[k](h_left)

      out = out + self.relu(v_feat + h_feat)
    return out