import numpy as np
import torch

def update_ema_variables(model, ema_model, alpha, global_step):
    """
    Mean Teacher 的核心：使用指数移动平均 (EMA) 更新教师模型的权重。
    Teacher_params = alpha * Teacher_params + (1 - alpha) * Student_params
    """
    # 动态调整 alpha (可选): 训练初期让 Teacher 快速跟上 Student
    # alpha = min(1 - 1 / (global_step + 1), alpha)
    
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def get_current_consistency_weight(epoch, max_epochs, consistency=0.1, consistency_rampup=20.0):
    """
    计算当前 Epoch 的一致性损失权重 (sigmoid ramp-up)。
    在训练初期，一致性权重为 0，逐渐增加到最大值。
    """
    return consistency * sigmoid_rampup(epoch, consistency_rampup)

def sigmoid_rampup(current, rampup_length):
    """Sigmoid 形状的爬坡函数"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

class ConsistencyLoss(torch.nn.Module):
    """
    一致性损失 (MSE Loss)，用于计算 Student 和 Teacher 输出的差异
    """
    def __init__(self):
        super().__init__()
        self.mse_loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, student_output, teacher_output):
        # Teacher 的输出通常经过 Softmax，所以这里计算 MSE
        return self.mse_loss(student_output, teacher_output)