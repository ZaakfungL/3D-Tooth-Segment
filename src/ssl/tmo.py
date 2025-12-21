import torch
from torch.optim import Optimizer
import math

class TMOAdamW(Optimizer):
    """
    TMO (Trusted Momentum Optimization) AdamW 优化器
    
    核心机制：
    1. Labeled Step: 仅使用有标签梯度更新动量，建立“可信方向” (Trusted Direction)。
    2. Unlabeled Step: 检查无标签梯度与可信方向的对齐度，进行门控筛选，然后完成参数更新。
    
    对应论文: "Hierarchical Cautious Optimization..." (Algorithm 1)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(TMOAdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step_labeled(self):
        """
        第一步：使用有标签梯度 (g_L) 更新动量，计算可信方向 (u_L)。
        注意：此步骤不更新参数权重 w，只更新动量状态 m_t, v_t。
        """
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                # g_L
                grad = p.grad 
                
                state = self.state[p]
                
                # 初始化状态
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format) # m_t
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format) # v_t

                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # --- 论文 Algorithm 1 Line 5-6 ---
                # 使用 g_L 更新动量
                # m_t = beta1 * m_{t-1} + (1-beta1) * g_L
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t = beta2 * v_{t-1} + (1-beta2) * g_L^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # --- 论文 Algorithm 1 Line 7-9 ---
                # 计算偏差修正后的 u_L_t (可信方向)
                # 注意：这里我们暂存 u_L，供下一步使用
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                u_L = (exp_avg / bias_correction1) / denom
                
                # 将可信方向 u_L 存入 state，供下一步对齐检查
                state['u_L'] = u_L.clone()

    @torch.no_grad()
    def step_unlabeled(self):
        """
        第二步：使用无标签梯度 (g_U)，执行对齐检查，并完成最终参数更新。
        """
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            beta1, beta2 = group['betas']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                # g_U
                grad_U = p.grad
                state = self.state[p]
                
                # 必须先调用 step_labeled
                if 'u_L' not in state:
                    raise RuntimeError("TMO Error: step_unlabeled called before step_labeled or state missing.")
                
                u_L = state['u_L']
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # --- 论文 Algorithm 1 Line 11: Alignment Gate ---
                # 检查 g_U 与 可信方向 u_L 是否一致
                # phi = I(u_L * g_U > 0)
                # 论文中使用 Hadamard product (element-wise)
                alignment_mask = (u_L * grad_U) > 0
                phi = alignment_mask.float()

                # --- 论文 Algorithm 1 Line 12-13: Refine Momentum ---
                # 仅将对齐部分的 g_U 融入动量
                # m_t = m_t + (1-beta1) * (g_U * phi)
                # 注意：是在 step_labeled 更新后的 m_t 基础上继续累加
                masked_g_U = grad_U * phi
                
                exp_avg.add_(masked_g_U, alpha=1 - beta1)
                exp_avg_sq.addcmul_(masked_g_U, masked_g_U, value=1 - beta2)

                # --- 论文 Algorithm 1 Line 14-16: Compute Final Update Direction u_t ---
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                u_t = (exp_avg / bias_correction1) / denom

                # --- 论文 Algorithm 1 Line 17: Normalized Gate ---
                # phi_bar = phi / max(mean(phi), eps)
                # 避免更新量因 masking 而变得过小
                phi_mean = phi.mean()
                phi_bar = phi / torch.max(phi_mean, torch.tensor(eps, device=phi.device))

                # --- 论文 Algorithm 1 Line 18: Final Parameter Update ---
                # w_t = w_{t-1} - lr * u_L - lr * (phi_bar * u_t)
                # 这里实现了：可信方向 + 门控后的修正方向
                
                # 1. 应用 Weight Decay (Decoupled like AdamW)
                if weight_decay > 0:
                    p.data.mul_(1 - lr * weight_decay)

                # 2. 应用梯度更新
                # update = u_L + phi_bar * u_t
                final_update = u_L.add(u_t * phi_bar)
                
                p.data.add_(final_update, alpha=-lr)
                
                # 清理临时状态以省显存
                del state['u_L']