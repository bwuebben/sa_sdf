from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class SASDFLoss(nn.Module):
    """
    Composite Loss: Pricing Error + Transaction Costs + CVaR
    Eq (46) 
    """
    def __init__(
        self,
        lambda_tc: float = 0.01,
        lambda_cvar: float = 0.5,
        confidence_level: float = 0.95
    ) -> None:
        super().__init__()
        self.lambda_tc: float = lambda_tc
        self.lambda_cvar: float = lambda_cvar
        self.tau: float = confidence_level
        self.epsilon: float = 1e-6  # For smooth absolute value [cite: 473]

    def smooth_abs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Differentiable approximation of |x|: sqrt(x^2 + eps)

        Args:
            x: Input tensor

        Returns:
            Smoothed absolute value of x
        """
        return torch.sqrt(x**2 + self.epsilon)

    def forward(
        self,
        weights_t: torch.Tensor,
        returns_t1: torch.Tensor,
        weights_prev_plus: torch.Tensor,
        costs_t: torch.Tensor,
        cvar_nu: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            weights_t: (batch, N) - Current portfolio weights w_t
            returns_t1: (batch, N) - Excess returns R_{t+1}
            weights_prev_plus: (batch, N) - Weights after drift w_{t-1}^+ [cite: 447]
            costs_t: (batch, N) - Transaction cost estimates c_{i,t}
            cvar_nu: (1,) - Learnable VaR threshold variable \nu [cite: 485]

        Returns:
            Tuple of (total_loss, metrics) where:
                total_loss: Scalar tensor with composite loss value
                metrics: Dictionary with individual loss components
        """
        # 1. Pricing Error (Maximize Sharpe / Minimize Deviation from 1)
        # SDF = 1 - w'R [cite: 95]
        port_ret = (weights_t * returns_t1).sum(dim=1)
        sdf = 1.0 - port_ret
        loss_price = torch.mean(sdf**2)
        
        # 2. Transaction Costs Penalty [cite: 449]
        # Delta w = w_t - w_{t-1}^+
        turnover = self.smooth_abs(weights_t - weights_prev_plus)
        tc_cost = (costs_t * turnover).sum(dim=1).mean()
        
        # 3. Tail Risk (CVaR) Penalty [cite: 485]
        # Variational formulation: nu + 1/(1-tau) * E[ (Loss - nu)+ ]
        # Loss l_t = - R_p
        losses = -port_ret
        
        # Softplus approximation for ReLU [cite: 491]
        hinge = F.softplus(losses - cvar_nu, beta=20)
        cvar_term = cvar_nu + (1.0 / (1.0 - self.tau)) * torch.mean(hinge)
        
        # Total Loss
        total_loss = loss_price + (self.lambda_tc * tc_cost) + (self.lambda_cvar * cvar_term)
        
        return total_loss, {
            "price_error": loss_price.item(),
            "tc_cost": tc_cost.item(),
            "cvar": cvar_term.item(),
            "avg_ret": port_ret.mean().item()
        }

