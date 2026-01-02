from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class TemporalStateModule(nn.Module):
    """
    Implements the Linear State-Space Model (SSM) for regime detection.
    Section 3.2 / Eq (11)-(12) / Appendix A.1
    """
    def __init__(self, num_chars: int, hidden_dim: int = 64, num_state_tokens: int = 8) -> None:
        super().__init__()
        self.D: int = num_chars
        self.K: int = num_state_tokens
        self.d_h: int = hidden_dim
        self.d_in: int = 4 * num_chars  # Summary stats: mean, std, q_low, q_high [cite: 234]

        # SSM Parameters: h_t = A*h_{t-1} + B*X_bar_t
        # A initialized to encourage persistence (Appendix A.1.1) [cite: 1310]
        self.A_log = nn.Parameter(torch.log(torch.rand(hidden_dim) * 0.09 + 0.9)) # diagonal A in (0.9, 0.99)
        self.B = nn.Linear(self.d_in, hidden_dim, bias=False)
        
        # Output projections: S_t = C*h_t + D*X_bar_t
        self.C = nn.Linear(hidden_dim, num_state_tokens * num_chars)
        self.D_proj = nn.Linear(self.d_in, num_state_tokens * num_chars)
        
        # Discretization step size (learnable) [cite: 1334]
        self.dt = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_summary: torch.Tensor, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_summary: (batch, 4*D) - Cross-sectional summary stats
        h_prev: (batch, d_h) - Hidden state from t-1

        Returns:
            Tuple of (s_t, h_t) where:
                s_t: (batch, K, D) - State tokens
                h_t: (batch, d_h) - Updated hidden state
        """
        # Discretize A (simplified diagonal approximation for stability)
        A = torch.exp(self.A_log) 
        
        # State update: h_t = A * h_{t-1} + B * x_sum
        h_t = A * h_prev + self.B(x_summary)
        
        # Generate State Tokens S_t: (batch, K, D)
        s_t = self.C(h_t) + self.D_proj(x_summary)
        s_t = rearrange(s_t, 'b (k d) -> b k d', k=self.K, d=self.D)
        
        return s_t, h_t

class SASDF(nn.Module):
    """
    Full State-Aware SDF Model.
    Integrates Temporal SSM + Cross-Sectional Transformer + Portfolio Layer.
    Section 3.1 [cite: 221]
    """
    def __init__(
        self,
        num_chars: int = 132,
        d_model: int = 132,
        nhead: int = 4,
        num_layers: int = 2,
        num_state_tokens: int = 8,
        leverage_scale: float = 1.0
    ) -> None:
        super().__init__()
        self.num_chars: int = num_chars
        self.leverage_scale: float = leverage_scale  # Kappa in Eq (31) [cite: 385]
        
        # 1. Temporal Module
        self.temporal_ssm = TemporalStateModule(num_chars, num_state_tokens=num_state_tokens)
        
        # 2. Transformer Encoder (Pre-LN) [cite: 1379]
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=4*d_model,
                                                   dropout=0.1, activation='gelu',
                                                   batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                                 enable_nested_tensor=False)
        
        # 3. Portfolio Layer (Readout) [cite: 224]
        self.readout = nn.Linear(d_model, 1, bias=False)

    def compute_summary_stats(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute mean, std, q25, q75 across cross-section [cite: 234]

        Args:
            x: (batch, N_assets, D) - Asset characteristics

        Returns:
            (batch, 4*D) - Concatenated summary statistics
        """
        # x: (batch, N_assets, D)
        mu = x.mean(dim=1)
        sigma = x.std(dim=1)
        q25 = torch.quantile(x, 0.25, dim=1)
        q75 = torch.quantile(x, 0.75, dim=1)
        return torch.cat([mu, sigma, q25, q75], dim=-1)

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_t: (batch, N_assets, D) - Asset characteristics at time t
            h_prev: (batch, hidden_dim) - SSM state from previous timestep
            mask: Optional attention mask for graph priors (Section 3.4)

        Returns:
            Tuple of (w_t, h_next) where:
                w_t: (batch, N_assets) - Portfolio weights
                h_next: (batch, hidden_dim) - Updated SSM state
        """
        batch_size, n_assets, _ = x_t.shape
        
        # --- Temporal Step ---
        x_summary = self.compute_summary_stats(x_t)
        
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.temporal_ssm.d_h, device=x_t.device)
            
        # Get regime tokens S_t
        s_t, h_next = self.temporal_ssm(x_summary, h_prev)
        
        # --- Augmentation Step ---
        # Concatenate assets X_t and state tokens S_t [cite: 290]
        # Input becomes (batch, N + K, D)
        augmented_input = torch.cat([x_t, s_t], dim=1)
        
        # --- Cross-Sectional Attention ---
        # Note: If implementing Graph Mask, pass src_mask here [cite: 342]
        hidden = self.transformer(augmented_input)
        
        # --- Portfolio Construction ---
        # Extract only asset representations (first N rows) [cite: 378]
        z_t = hidden[:, :n_assets, :]
        
        # Linear projection to raw weights
        raw_weights = self.readout(z_t).squeeze(-1) # (batch, N)
        
        # Weight Normalization (L1 norm) [cite: 385]
        norm = torch.sum(torch.abs(raw_weights), dim=1, keepdim=True) + 1e-6
        w_t = (raw_weights / norm) * self.leverage_scale
        
        return w_t, h_next

