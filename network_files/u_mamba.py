import torch
from torch import nn
from torch.nn import functional as F


import os
import sys


moudle_dir=os.path.abspath(os.path.join(os.path.dirname(__file__),'../VMamba'))


sys.path.append(moudle_dir)


from vmamba import VSSBlock


# ************************************************
# ********* 2. U-Mamba 核心构建块 *********
# ************************************************

# CNN + Mamba 混合块 (取代 doubleConv)
class MambaConvBlock(nn.Module):
   def __init__(self, in_channels, out_channels, dropout_prob, mid_channels=None):
      super().__init__()
      if mid_channels is None:
         mid_channels = out_channels
         
      # 1. CNN 部分 (Local Feature Extraction)
      # 沿用原 doubleConv 的逻辑
      cnn_layers = []
      cnn_layers.append(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False))
      cnn_layers.append(nn.BatchNorm2d(mid_channels))
      cnn_layers.append(nn.ReLU(inplace=True))
      
      if dropout_prob > 0:
            cnn_layers.append(nn.Dropout2d(p=dropout_prob))
            
      cnn_layers.append(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False))
      cnn_layers.append(nn.BatchNorm2d(out_channels))
      cnn_layers.append(nn.ReLU(inplace=True))
      self.cnn_block = nn.Sequential(*cnn_layers)
      
      # 2. Mamba 部分 (Global Context Modeling)
      # 在 CNN 块的输出维度上应用 VSSBlock
      # 这里 channel_first=True，因为输入是 (B, C, H, W)
      # FORWARD_TYPES 字典中存在的键。
      self.mamba_block = VSSBlock(hidden_dim=out_channels,channel_first=True,forward_type='v03') 

   def forward(self, x):
      # 1. CNN 提取局部特征
      x_cnn = self.cnn_block(x)
      
      # 2. Mamba 增强全局上下文
      # 注意：这里是残差增强，Mamba 块的输出直接加到 CNN 块的输出上
      x_mamba_enhanced = self.mamba_block(x_cnn)
      
      return x_mamba_enhanced

# 下采样 (Down Layer) - 采用 MambaConvBlock
def down_mamba(in_channels, out_channels, dropout_prob):
   # 池化 + MambaConvBlock
   layer = []
   layer.append(nn.MaxPool2d(2, stride=2))
   layer.append(MambaConvBlock(in_channels, out_channels, dropout_prob))
   return nn.Sequential(*layer)


# 上采样 + 拼接 (Up Layer) - 采用 MambaConvBlock
class Up_Mamba(nn.Module):
   def __init__(self, in_channels, out_channels, bilinear=True, dropout_prob=0.0):
      super(Up_Mamba, self).__init__()
      
      if bilinear:
         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
         # 拼接后输入通道数仍为 in_channels
         self.conv = MambaConvBlock(in_channels, out_channels, dropout_prob, in_channels // 2) 
      else:
         # 对于转置卷积，其输入通道数是上一层(更深层)的输出通道数，输出通道数通常是当前层的一半
         # 例如 up1: in_channels=1024, out_channels=512。上一层输出是512。
         # 所以转置卷积的输入是512，输出是256
         prev_level_channels = in_channels - (out_channels * 2) # 这是来自跳跃连接的通道数
         if prev_level_channels < 0: # 瓶颈层特殊处理
             prev_level_channels = in_channels // 2
         
         self.up = nn.ConvTranspose2d(prev_level_channels, prev_level_channels // 2, kernel_size=2, stride=2)
         self.conv = MambaConvBlock(in_channels - prev_level_channels + (prev_level_channels // 2), out_channels, dropout_prob)


   def forward(self, x1, x2):
      # x1 是来自瓶颈层或下层解码器的特征
      x1 = self.up(x1)
      
      # 调整 x1 和 x2 的尺寸匹配
      # 如果尺寸不完全匹配 (如奇数尺寸)，需要裁剪 x2
      diffY = x2.size()[2] - x1.size()[2]
      diffX = x2.size()[3] - x1.size()[3]
      x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                     diffY // 2, diffY - diffY // 2])
      
      # 拼接 (Skip Connection)
      x = torch.cat([x1, x2], dim=1)
      
      # 经历 Mamba 增强双卷积
      x = self.conv(x)
      return x


# ************************************************
# ********* 3. 最终 U-Mamba 网络结构 *********
# ************************************************

class U_Mamba_net(nn.Module):
   '''
   U-Net 结合 Mamba 模块的混合结构 (仿 U-Mamba/VM-UNet 思想)
   '''
   def __init__(self, in_channels, out_channels, bilinear=True, base_channel=32, dropout_prob=0.0):
      super(U_Mamba_net, self).__init__()
      self.in_channels = in_channels
      self.out_channels = out_channels
      self.bilinear = bilinear

      # 1. 输入层 (使用 MambaConvBlock)
      self.in_conv = MambaConvBlock(self.in_channels, base_channel, dropout_prob)
      
      # 2. 编码器 (Down Layers)
      self.down1 = down_mamba(base_channel, base_channel*2, dropout_prob) 
      self.down2 = down_mamba(base_channel*2, base_channel*4, dropout_prob)
      self.down3 = down_mamba(base_channel*4, base_channel*8, dropout_prob)
      
      # 3. 瓶颈层/最深层
      factor = 2 if self.bilinear else 1
      self.down4 = down_mamba(base_channel*8, base_channel*16 // factor, dropout_prob) 

      # 4. 解码器 (Up Layers)
      self.up1 = Up_Mamba(base_channel*16 , base_channel*8 // factor, self.bilinear, dropout_prob) 
      self.up2 = Up_Mamba(base_channel*8 , base_channel*4 // factor, self.bilinear, dropout_prob)
      self.up3 = Up_Mamba(base_channel*4 , base_channel*2 // factor, self.bilinear, dropout_prob)
      self.up4 = Up_Mamba(base_channel*2 , base_channel, self.bilinear, dropout_prob)
      
      # 5. 输出层
      self.out = nn.Conv2d(in_channels=base_channel, out_channels=self.out_channels, kernel_size=1)

   def forward(self, x):
      # 编码器路径
      x1 = self.in_conv(x) # 64
      x2 = self.down1(x1)  # 128
      x3 = self.down2(x2)  # 256
      x4 = self.down3(x3)  # 512
      x5 = self.down4(x4)  # 512 (Bottleneck)
      
      # 解码器路径 (跳跃连接)
      x = self.up1(x5, x4) # 1024 -> 512
      x = self.up2(x, x3)  # 512 -> 256
      x = self.up3(x, x2)  # 256 -> 128
      x = self.up4(x, x1)  # 128 -> 64
      
      out = self.out(x)
      return {'out': out}