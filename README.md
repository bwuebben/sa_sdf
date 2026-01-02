# State-Aware Stochastic Discount Factors (SA-SDF)

PyTorch implementation of **State-Aware Stochastic Discount Factors**, a deep learning model for portfolio optimization that combines temporal regime detection with cross-sectional asset analysis.

**Paper**: *State-Aware Stochastic Discount Factors* by Bernd J. Wuebben (AllianceBernstein, 2026)

## Overview

The SA-SDF addresses three key limitations of existing machine learning approaches to asset pricing:

1. **Temporal Myopia**: Traditional models treat each month independently without capturing persistent market regimes
2. **Implementation Misalignment**: Training objectives ignore transaction costs and tail risks
3. **Unstructured Information Flow**: Cross-asset attention lacks economic structure

### Key Innovations

**1. Temporal State Module**
- Linear State-Space Model (SSM) for regime detection with theoretically unbounded memory
- Learns K regime tokens that summarize market conditions (stress, sentiment, business cycle)
- Effective memory spans up to 11 months (largest eigenvalue: 0.94)

**2. State-Conditioned Cross-Sectional Attention**
- Transformer architecture processes augmented input: [Asset Characteristics; State Tokens]
- Attention patterns modulate based on learned market regimes
- During stress periods: attention becomes more concentrated on liquid, within-industry assets

**3. Cost-Aware Training Objective**
- Direct incorporation of transaction costs (turnover penalty)
- CVaR tail-risk penalty for downside protection
- Learns representations that are inherently low-turnover and tail-robust

## Architecture

```
Input: X_t (N_assets × D_chars)
   ↓
   ├─→ Summary Stats (mean, std, q25, q75)
   │      ↓
   │   Temporal SSM: h_t = A·h_{t-1} + B·X̄_t
   │      ↓
   │   State Tokens S_t (K × D)
   │      ↓
   └─→ [X_t; S_t] (Augmented Input)
          ↓
   Transformer Encoder (10 layers)
          ↓
   Portfolio Layer: w_t = norm(Z_t·λ)
```

### Model Components

- **TemporalStateModule**: SSM with selective gating and learnable transition dynamics
- **SASDF**: Full model integrating temporal, cross-sectional, and portfolio layers
- **SASDFLoss**: Composite loss = Pricing Error + λ_TC·TC + λ_CVaR·CVaR

## Installation

```bash
# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.1+
- NumPy, Pandas, SciPy
- einops (tensor operations)
- tqdm (progress bars)

## Usage

### Basic Training

```bash
python train.py
```

This runs the demo training script with mock data:
- 120 months of synthetic data
- 50 assets, 132 characteristics
- 5 training epochs
- Outputs loss and CVaR metrics

### Understanding the Code

**Model Initialization:**
```python
from sa_sdf.model import SASDF
from sa_sdf.loss import SASDFLoss

# Initialize model
model = SASDF(
    num_chars=132,      # Number of characteristics
    num_state_tokens=8, # Regime tokens (K)
    d_model=132,        # Transformer dimension
    nhead=4,            # Attention heads
    num_layers=10       # Transformer depth
)

# Initialize loss function
loss_fn = SASDFLoss(
    lambda_tc=0.01,     # Transaction cost penalty
    lambda_cvar=0.5,    # Tail risk penalty
    confidence_level=0.95
)

# CVaR parameter (must be optimized!)
cvar_nu = torch.tensor(0.0, requires_grad=True)
```

**Training Loop:**
```python
# Sequential processing required!
h_state = None  # Hidden state persists across time
prev_weights = torch.zeros(1, N_assets)

for t in range(T - 1):
    # Forward pass with state
    weights_t, h_state_new = model(x_t, h_state)
    h_state = h_state_new.detach()  # Truncated BPTT

    # Compute loss with transaction costs
    loss, metrics = loss_fn(
        weights_t, returns_t1,
        prev_weights, costs_t, cvar_nu
    )

    # Optimize both model and CVaR threshold
    optimizer.step()
    prev_weights = weights_t.detach()
```

## Key Results (from Paper)

### Performance Improvements
- **Net Sharpe Ratio**: 1.43 vs. 1.15 (KKMU Transformer) — 24% improvement
- **Post-2002 Period**: 1.24 vs. 0.78 — 59% improvement (where many ML strategies decay)
- **Hansen-Jagannathan Distance**: 0.11 vs. 0.15 — 27% reduction in pricing errors

### Risk Metrics
- **Maximum Drawdown**: 26.8% vs. 33.4% — 20% reduction
- **Turnover**: 38.6% vs. 71.2% monthly — 46% lower trading costs
- **CVaR (95%)**: 8.14% vs. 9.86% — Better tail protection

### Crisis Performance
Average drawdown improvement across 9 major crises:
- 2008 Financial Crisis: -21.4% vs. -31.2% (+9.8pp)
- 2020 COVID Crash: -11.2% vs. -18.4% (+7.2pp)
- Recovery time: 7.8 months vs. 13.7 months on average

## Model Features

### Learned State Tokens
The 8 state tokens capture economically meaningful regimes:
- **S₁ (Stress)**: Correlates with VIX (0.72), funding spreads (0.58), cross-sectional dispersion (0.64)
- **S₂ (Sentiment)**: Correlates with Baker-Wurgler sentiment (0.42)
- **S₃ (Growth)**: Correlates with CFNAI business cycle indicator (0.32)
- **S₄-S₈**: Capture residual dynamics

### Attention Dynamics
- **Calm Markets**: Entropy = 7.21, diffuse attention across assets
- **Stress Markets**: Entropy = 6.14, concentrated on liquid/large-cap stocks
- **Within-Industry**: 18.2% → 34.8% during high-VIX periods

### Graph Priors (Optional)
- Industry classifications (Fama-French 48)
- Supply chain links (customer-supplier relationships)
- Analyst coverage overlap
- Learnable gate adapts reliance on economic structure

## Data Format

### Expected Inputs
- **Characteristics (X_t)**: (T, N_assets, D_chars) — Rank-transformed to [-0.5, 0.5]
- **Returns (R_t)**: (T, N_assets) — Excess returns
- **Costs (C_t)**: (T, N_assets) — Transaction cost estimates (bid-ask spread)

### Key Constraints
1. **Sequential Processing**: Data must be in temporal order (no random shuffling)
2. **Hidden State Persistence**: h_state carries information across time steps
3. **Detached States**: Use `.detach()` for truncated BPTT to manage memory

## Architecture Details

### Hyperparameters (from paper)
| Parameter | Value | Description |
|-----------|-------|-------------|
| D (num_chars) | 132 | Number of asset characteristics |
| K (state_tokens) | 8 | Number of regime tokens |
| d_h (hidden_dim) | 64 | SSM hidden state dimension |
| L_blocks | 10 | Transformer encoder layers |
| H (nhead) | 8 | Multi-head attention heads |
| d_ff | 4×D | Feed-forward dimension |
| λ_TC | 0.01 | Transaction cost penalty |
| λ_CVaR | 0.5 | Tail risk penalty |
| τ (CVaR) | 0.95 | Confidence level |

### Training Configuration
- **Optimizer**: AdamW (lr=5e-5, weight_decay=1e-3)
- **Learning Rate**: Linear warmup (5 epochs) + cosine decay
- **Gradient Clipping**: max_norm=1.0
- **Training Window**: 60 months rolling
- **Ensemble**: 10 models with different seeds

## File Structure

```
sa_sdf/
├── README.md           # This file
├── CLAUDE.md          # Development guide for Claude Code
├── requirements.txt   # Python dependencies
├── train.py           # Training script with mock data
├── docs/
│   └── paper.pdf     # Research paper
└── sa_sdf/
    ├── model.py      # SASDF and TemporalStateModule
    └── loss.py       # SASDFLoss (pricing + costs + CVaR)
```

## Important Implementation Notes

### Critical Requirements

1. **Optimize CVaR Parameter**
   ```python
   # MUST add cvar_nu to optimizer!
   optimizer = optim.AdamW(
       list(model.parameters()) + [cvar_nu],
       lr=5e-5
   )
   ```

2. **Sequential Time Processing**
   - Do NOT shuffle data across time
   - Hidden state must persist: `h_state = h_new.detach()`
   - Process "all assets for single month" per batch

3. **Weight Drift Calculation**
   ```python
   # Account for passive drift before computing turnover
   w_prev_plus = w_prev * (1 + r_t) / sum(w_prev * (1 + r_t))
   turnover = sum(|w_t - w_prev_plus| * costs)
   ```

## Citation

```bibtex
@article{wuebben2026sasdf,
  title={State-Aware Stochastic Discount Factors},
  author={Wuebben, Bernd J.},
  journal={Working Paper},
  institution={AllianceBernstein},
  year={2026}
}
```

## References

Key papers that informed this work:
- **Kelly, Kuznetsov, Malamud & Xu (2024)**: Transformer-based SDFs
- **Gu, Kelly & Xiu (2020)**: Neural network asset pricing
- **Gu, Goel & Ré (2022)**: Structured state-space models
- **Jensen, Kelly & Pedersen (2023)**: 132 characteristic dataset
- **Campbell & Cochrane (1999)**: Habit formation and time-varying risk premia

## License

Research implementation for academic and educational purposes.

## Contact

For questions about the implementation, see the paper: `docs/paper.pdf`

---

**Note**: This implementation uses mock data for demonstration. Real-world application requires:
- Historical stock characteristics (132 dimensions)
- Return data from CRSP
- Transaction cost estimates (TAQ or Corwin-Schultz)
- Proper out-of-sample backtesting with rolling windows
