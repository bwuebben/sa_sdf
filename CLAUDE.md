# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch implementation of **SA-SDF** (State-Aware Stochastic Discount Factor), a deep learning model for portfolio optimization that combines temporal regime detection with cross-sectional asset analysis.

The implementation follows a research paper (see `docs/paper.pdf`). Code comments include citation markers (e.g., `[cite: 573]`) that reference specific sections or equations in the paper.

## Development Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv env
source env/bin/activate  # On macOS/Linux
# env\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Training
```bash
# Run full training loop
python train.py
```

Note: There are currently no test files, linting configurations, or additional build scripts in this repository.

## Architecture

The SA-SDF model consists of three main components that process data sequentially:

### 1. Temporal State Module (`TemporalStateModule` in `model.py`)
- **Purpose**: Regime detection across time using a Linear State-Space Model (SSM)
- **Input**: Cross-sectional summary statistics (mean, std, q25, q75) of asset characteristics
- **Output**: K state tokens that capture current market regime
- **Key Features**:
  - Maintains hidden state `h_t` that persists across time steps
  - SSM update: `h_t = A * h_{t-1} + B * x_summary`
  - State tokens: `S_t = C * h_t + D * x_summary`
  - Diagonal matrix A initialized in [0.9, 0.99] for persistence
  - Learnable discretization timestep `dt`

### 2. Cross-Sectional Transformer (`transformer` in `SASDF`)
- **Purpose**: Model interactions between assets and regime states
- **Input**: Augmented sequence of [N assets + K state tokens]
- **Architecture**: Pre-LN Transformer Encoder with GELU activation
- **Mechanism**:
  - State tokens are concatenated with asset embeddings
  - Self-attention captures both asset-asset and asset-regime interactions
  - Only asset representations (first N tokens) are used for portfolio construction

### 3. Portfolio Layer (`readout` in `SASDF`)
- **Purpose**: Generate portfolio weights from asset representations
- **Process**:
  - Linear projection: `raw_weights = readout(z_t)`
  - L1 normalization: `w_t = (raw_weights / ||raw_weights||_1) * leverage_scale`
- **Output**: Portfolio weights summing to `leverage_scale` (default 1.0)

## Loss Function

The `SASDFLoss` combines three objectives (see Eq 46 in paper):

1. **Pricing Error**: Minimizes deviation of SDF from 1
   - SDF = 1 - w'R (portfolio return)
   - Loss: E[(SDF)^2]

2. **Transaction Costs** (λ_tc = 0.01):
   - Penalizes turnover: sum of |w_t - w_{t-1}^+| weighted by cost estimates
   - Uses smooth absolute value for differentiability

3. **CVaR Tail Risk** (λ_cvar = 0.5):
   - Variational CVaR formulation with learnable VaR threshold `nu`
   - Loss: nu + 1/(1-τ) * E[max(0, -R_p - nu)]
   - Uses softplus approximation for smooth gradients

**Critical**: The `cvar_nu` parameter must be added to the optimizer alongside model parameters.

## Training Procedure

The training loop in `train.py` implements Algorithm 1 from the paper with several important constraints:

### Sequential Processing
- **Data must be processed in temporal order** (not randomly shuffled)
- Batching is done per time step: all assets for a single month
- Hidden state `h_state` persists across time steps within an epoch

### State Management
```python
# Hidden state carries temporal memory
weights_t, h_state_new = model(x_t, h_state)
h_state = h_state_new.detach()  # Truncated BPTT
```

### Weight Drift Calculation
- `w_{t-1}^+` represents weights after market drift before rebalancing
- Currently simplified; full implementation should account for returns:
  ```python
  w_prev_plus = w_prev * (1 + r_t) / sum(w_prev * (1 + r_t))
  ```

### Optimization
- Optimizer: AdamW with weight decay 1e-3
- Learning rate: 5e-5
- Gradient clipping: max_norm=1.0
- Optimizes both `model.parameters()` and `cvar_nu`

## Key Implementation Details

### Data Format
- **Characteristics** (X_t): (T, N_assets, N_chars) - Rank-transformed to [-0.5, 0.5]
- **Returns** (R_t): (T, N_assets) - Excess returns for next period
- **Costs** (C_t): (T, N_assets) - Transaction cost estimates (e.g., bid-ask spread)

### Time Indexing Convention
- Input characteristics at time t: `x_t`
- Output weights at time t: `w_t`
- Returns for period t→t+1: `r_{t+1}`
- Loss at time t uses `(x_t, w_t, r_{t+1})`

### Module Import
Note: The `sa_sdf/` directory lacks an `__init__.py` file. Imports work via:
```python
from sa_sdf.model import SASDF
from sa_sdf.loss import SASDFLoss
```
This relies on the parent directory being in PYTHONPATH.

## Paper Citations

Throughout the code, you'll see markers like `[cite: 573]`. These reference:
- Specific equations (e.g., Eq 11, Eq 46)
- Sections (e.g., Section 3.2, Appendix A.1)
- Page numbers or paragraph numbers in `docs/paper.pdf`

When modifying the implementation, consult these citations to ensure alignment with the paper's methodology.
