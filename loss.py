import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridLoss(nn.Module):
    def __init__(self, weights, alpha=0.5, gamma=2.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weights, reduction='none')
        self.weights = weights
        self.alpha = alpha
        self.gamma = gamma
        self.eps = 1e-7  # 防止数值溢出的小常数

    def forward(self, logits, labels):
        # 1. 对logits进行clipping防止极端值
        logits = torch.clamp(logits, -100, 100)

        # 2. 计算CrossEntropy Loss
        ce_loss = self.ce(logits, labels)

        # 3. 计算Focal Loss，添加数值稳定性处理
        probs = F.softmax(logits, dim=-1)
        probs = torch.clamp(probs, self.eps, 1.0 - self.eps)  # 裁剪概率值

        # 4. 获取目标类别的概率
        p_t = probs[range(len(labels)), labels]

        # 5. 计算focal loss，添加防护措施
        focal_term = (1 - p_t) ** self.gamma
        focal_term = torch.clamp(focal_term, 0, 16)  # 防止指数项过大
        focal_loss = -focal_term * torch.log(p_t)

        if self.weights is not None:
            focal_loss = focal_loss * self.weights[labels]

        # 6. 合并损失
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * focal_loss

        # 7. 检查并处理无效值
        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
            print("Warning: NaN or Inf detected in loss calculation")
            print(f"CE Loss: {ce_loss.mean()}")
            print(f"Focal Loss: {focal_loss.mean()}")
            total_loss = torch.where(torch.isnan(total_loss) | torch.isinf(total_loss),
                                     torch.zeros_like(total_loss),
                                     total_loss)

        return total_loss.mean()