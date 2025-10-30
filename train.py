# from network_files.u_net import U_net
from network_files.u_mamba import U_Mamba_net

from utils import transforms as T
from My_Dataset import My_Dataset
from utils import Loss

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim

import os

# 验证的预处理方法获取
class SegmentationPresetEval:
   def __init__(self, base_size, crop_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
      
      # 验证模式下的变换序列（确保 H/W 尺寸固定为 crop_size）：
      trans = [
         # 1. 确定性缩放：将图像最短边缩放到 base_size，保证 CenterCrop 能裁剪出足够的区域
         #    注意：这里使用 T.RandomResize(size, size) 来实现确定性缩放
         T.RandomResize(base_size, base_size), 
         
         # 2. 中心裁剪：**关键步骤**，裁剪出模型的固定输入尺寸 (crop_size x crop_size)
         T.CenterCrop(crop_size),
         
         # 3. 转换为 Tensor
         T.ToTensor(),
         
         # 4. 归一化
         T.Normalize(mean=mean, std=std),
      ]
      
      # 使用自定义的 Compose 类来组合这些变换
      self.transforms = T.Compose(trans)

   def __call__(self, img, target):
      return self.transforms(img, target)
        
# 训练的预处理方法获取
class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        # 初始化
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        # 随机裁剪
        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)

# 获取预处理方法
def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
   base_size = 565
   crop_size = 480

   if train:
      # 训练模式获取的预处理方法
      return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
   # else:
      
   #    return T.Compose([
   #       T.Resize(base_size),
   #       T.CenterCrop(crop_size),
   #       T.ToTensor(),
   #       T.Normalize(mean=mean, std=std),
   #    ])
   else:
      # 验证模式：使用确定性的固定尺寸操作
      return SegmentationPresetEval(base_size, crop_size, mean=mean, std=std)


      
# def main():
#    # 设置基本参数信息
#    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#    print(torch.cuda.is_available())
#    # 如果GPU不够用，可以用cpu
# #  device = torch.device('cpu')

#    #超参数设置
#    batch_size= 4
#    epoch = 20
#    num_classes = 2 # 1(object)+1(background)
#    if num_classes == 2:
#       # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
#       loss_weight = torch.as_tensor([1.0, 2.0], device=device)
#    else:
#       loss_weight = None
      
#    # 加载数据
#    mean = (0.709, 0.381, 0.224)
#    std = (0.127, 0.079, 0.043)
#    train_dataset = My_Dataset('./',train=True,transforms=get_transform(train=True, mean=mean, std=std))
#    val_dataset = My_Dataset('./',train=False,transforms=get_transform(train=False, mean=mean, std=std))
#    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
#    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=True)
   
   
#    # 创建模型
#    model = U_net(3,num_classes) # 输入通道数，输出通道数
#    model.to(device)
#    # 定义优化器
#    params = [p for p in model.parameters() if p.requires_grad] # 定义需要优化的参数
#    sgd = optim.SGD(params,lr=0.01,momentum=0.9,weight_decay=1e-4)
#    # 开始训练
#    model.train()
#    for e in range(epoch):
#       loss_temp = 0
#       for i,(image,mask) in enumerate(train_loader):
#          image,mask = image.to(device),mask.to(device)
#          output = model(image)
         
#          loss = Loss.criterion(output, mask, loss_weight, num_classes=num_classes, ignore_index=255)
#          loss_temp += loss.item()
#          sgd.zero_grad()
#          loss.backward()
#          #更新参数
#          sgd.step()
#    #   print(f'第{e+1}个epoch,平均损失loss={loss_temp/(i+1)}')
#       # 验证集评估
#       dice_val = evaluate_dice(model, val_loader, device)
      
      
#       print(f'第{e+1}个epoch, 平均损失loss={loss_temp/(i+1)},验证集平均 Dice={dice_val:.4f}')
      
#       save_model(model)
      
#    # 保存权重
#    save_model(model)
# #  save_dir = 'save_weights'
# #  os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在就自动创建
# #  name = os.path.join(save_dir, 'u_net.pth')
# #  torch.save(model.state_dict(), name)



import matplotlib.pyplot as plt

def main():
   # 设置基本参数信息
   device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
   print(torch.cuda.is_available())

   #超参数设置
   batch_size= 2
   epoch = 20
   num_classes = 2
   if num_classes == 2:
      loss_weight = torch.as_tensor([1.0, 2.0], device=device)
   else:
      loss_weight = None
      
   # 加载数据
   mean = (0.709, 0.381, 0.224)
   std = (0.127, 0.079, 0.043)
   train_dataset = My_Dataset('./',train=True,transforms=get_transform(train=True, mean=mean, std=std))
   val_dataset = My_Dataset('./',train=False,transforms=get_transform(train=False, mean=mean, std=std))
   train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
   val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=True)
   
   # 创建模型
   # model = U_net(3,num_classes)
   model = U_Mamba_net(3,num_classes)
   model.to(device)
   params = [p for p in model.parameters() if p.requires_grad]
   sgd = optim.SGD(params,lr=0.01,momentum=0.9,weight_decay=1e-4)

   # 用于保存训练和验证指标
   train_loss_list = []
   val_dice_list = []

   model.train()
   for e in range(epoch):
      loss_temp = 0
      for i,(image,mask) in enumerate(train_loader):
         image,mask = image.to(device),mask.to(device)
         output = model(image)
         
         loss = Loss.criterion(output, mask, loss_weight, num_classes=num_classes, ignore_index=255)
         loss_temp += loss.item()
         sgd.zero_grad()
         loss.backward()
         sgd.step()
         
      avg_loss = loss_temp/(i+1)
      train_loss_list.append(avg_loss)

      # 验证集评估
      dice_val = evaluate_dice(model, val_loader, device)
      val_dice_list.append(dice_val)
      
      print(f'第{e+1}个epoch, 平均损失loss={avg_loss:.4f}, 验证集平均 Dice={dice_val:.4f}')
      
      save_model(model)

   # 训练结束后画曲线
   plt.figure(figsize=(10,4))
   plt.subplot(1,2,1)
   plt.plot(range(1, epoch+1), train_loss_list, marker='o', label='Train Loss')
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.title('Training Loss Curve')
   plt.grid(True)
   plt.legend()

   plt.subplot(1,2,2)
   plt.plot(range(1, epoch+1), val_dice_list, marker='o', color='orange', label='Validation Dice')
   plt.xlabel('Epoch')
   plt.ylabel('Dice')
   plt.title('Validation Dice Curve')
   plt.grid(True)
   plt.legend()

   plt.tight_layout()
   plt.show()

   # 保存最终模型
   save_model(model)



def save_model(model):
       
    save_dir = 'save_weights'
    os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在就自动创建
    name = os.path.join(save_dir, 'u_net.pth')
    torch.save(model.state_dict(), name)

def evaluate_dice(model, dataloader, device):
   model.eval()
   dice_total = 0
   with torch.no_grad():
      for images, masks in dataloader:
         images, masks = images.to(device), masks.to(device)
         outputs = model(images)
         
         if isinstance(outputs, dict):
               outputs = outputs['out']
               
         # 二值化预测（pred的值为 0 或 1）
         pred = torch.argmax(outputs, dim=1)
         
         # === 关键修正：确保 masks 的值也是 0 或 1 ===
         # 如果您的 masks 包含 255（背景或忽略），您需要先处理它。
         # 最常见的情况是：前景类别索引是 1。我们只计算类别 1 的 Dice。
         
         # 1. 确保 masks 是整数类型 (例如 torch.int64) 
         masks = masks.long() 
         
         # 2. 如果 masks 的值是 0/255，将其转换为 0/1 (如果您的前景是 255)
         # if masks.max().item() > 1:
         #     # 假设 255 是前景，转换为 1；其他都是 0。
         #     masks = (masks > 0).long()
         
         # 3. 如果是多分类，但这里只计算前景（类别 1）的 Dice
         # 对于二分类，pred 和 masks 应该都只包含 0 和 1。
         # 如果 pred 的 max() > 1，说明是多分类，但您的计算方式是错误的！
         # 假设您只关注类别 1：
         # pred_fg = (pred == 1).long()
         # masks_fg = (masks == 1).long()
         
         # 在二分类场景下，我们假设 pred 和 masks 已经被正确地二值化为 0 和 1
         
         # Dice系数计算
         intersection = (pred * masks).sum()
         dice = (2. * intersection + 1e-6) / (pred.sum() + masks.sum() + 1e-6)
         dice_total += dice.item()
         
   model.train()
   return dice_total / len(dataloader)
 
# def evaluate_dice(model, dataloader, device):
#     model.eval()
#     dice_total = 0
#     with torch.no_grad():
#         for images, masks in dataloader:
#             images, masks = images.to(device), masks.to(device)
#             outputs = model(images)
            
#             if isinstance(outputs, dict):
#                 outputs = outputs['out']
                
                
#             # 二值化预测（针对二分类分割）
#             pred = torch.argmax(outputs, dim=1)
#             # Dice系数计算
#             intersection = (pred * masks).sum()
#             dice = (2. * intersection + 1e-6) / (pred.sum() + masks.sum() + 1e-6)
#             dice_total += dice.item()
#     model.train()
#     return dice_total / len(dataloader)

if __name__ == '__main__':
    main()
