import torch
from torch import nn

# dice : 度量两个集合的相似性

# loss
def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
    losses = {}
    for name, x in inputs.items():
        # x = torch.Size([4, 2, 480, 480]) ，4为batch，2表示输出通道数，480*480表示和原图大小一样
        # target = torch.Size([4, 480, 480]) 真实mask值
        # ignore_index = 255，表示忽略那些不感兴趣的部分
        loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
        if dice is True:
            # 计算每个类别的dice（背景、前景），然后求均值
            # 因此需要为每个类别构建一个target
            dice_target = build_target(target, num_classes, ignore_index)
            loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
        losses[name] = loss

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def build_target(target: torch.Tensor, num_classes: int = 2, ignore_index: int = -100):
    """build target for dice coefficient"""
    dice_target = target.clone()
    if ignore_index >= 0:
        # 将255的区域设置为0，因为不感兴趣的不需要计算dice，因此先设置为0
        ignore_mask = torch.eq(target, ignore_index)
        dice_target[ignore_mask] = 0
        # target转为one-hot编码形式 [1 0]表示一个类别 and [0 1]表示一个类别
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()
        # 将255的值又填充回去
        dice_target[ignore_mask] = ignore_index
    else:
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()

    # [N, H, W] -> [N, H, W, C]
    return dice_target.permute(0, 3, 1, 2)


def dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    # 计算一个batch中所有图片某个类别的dice_coefficient
    d = 0.
    batch_size = x.shape[0]
    # 遍历每张图片
    for i in range(batch_size):
        x_i = x[i].reshape(-1)
        t_i = target[i].reshape(-1)
        if ignore_index >= 0:
            # 找出mask中不为ignore_index的区域
            roi_mask = torch.ne(t_i, ignore_index)
            x_i = x_i[roi_mask]
            t_i = t_i[roi_mask]
        inter = torch.dot(x_i, t_i)
        sets_sum = torch.sum(x_i) + torch.sum(t_i)
        if sets_sum == 0:
            sets_sum = 2 * inter

        # 计算dice
        d += (2 * inter + epsilon) / (sets_sum + epsilon)

    return d / batch_size


def multiclass_dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    """Average of Dice coefficient for all classes"""
    dice = 0.
    # 遍历每个channel，计算每个类别的dice
    for channel in range(x.shape[1]):
        dice += dice_coeff(x[:, channel, ...], target[:, channel, ...], ignore_index, epsilon)

    # 求均值
    return dice / x.shape[1]


def dice_loss(x: torch.Tensor, target: torch.Tensor, multiclass: bool = False, ignore_index: int = -100):
    # 在channel方向做softmax
    x = nn.functional.softmax(x, dim=1)
    # 选择采用的方法
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(x, target, ignore_index=ignore_index)
