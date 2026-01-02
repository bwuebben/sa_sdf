from typing import Optional
import torch
import torch.optim as optim
from sa_sdf.model import SASDF
from sa_sdf.loss import SASDFLoss


def train_sa_sdf() -> None:
    # --- Hyperparameters [cite: 573] ---
    BATCH_SIZE: int = 1  # Process sequentially for time-series validity
    SEQ_LEN: int = 60  # Rolling window size
    N_EPOCHS: int = 5
    LR: float = 5e-5
    N_ASSETS: int = 50  # Mock size
    N_CHARS: int = 132
    
    # --- Mock Data Generation ---
    # T months of data
    T: int = 120
    # Characteristics (Rank Transformed [-0.5, 0.5]) [cite: 608]
    data_X: torch.Tensor = torch.rand(T, N_ASSETS, N_CHARS) - 0.5
    # Excess Returns
    data_R: torch.Tensor = torch.randn(T, N_ASSETS) * 0.05
    # Transaction Costs (e.g., bid-ask spread)
    data_C: torch.Tensor = torch.rand(T, N_ASSETS) * 0.001

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Model
    model: SASDF = SASDF(num_chars=N_CHARS, num_state_tokens=8).to(device)

    # Initialize Loss & Auxiliary Param \nu for CVaR [cite: 485]
    cvar_nu: torch.Tensor = torch.tensor(0.0, requires_grad=True, device=device)
    loss_fn: SASDFLoss = SASDFLoss(lambda_tc=0.01, lambda_cvar=0.5)

    # Optimizer (AdamW) [cite: 518]
    # Note: Optimize model parameters AND cvar_nu
    optimizer: optim.AdamW = optim.AdamW(list(model.parameters()) + [cvar_nu], lr=LR, weight_decay=1e-3)
    
    print("Starting Training...")
    
    # Training Loop (Algorithm 1)
    model.train()

    # Hidden state persists across time steps in the sequence
    h_state: Optional[torch.Tensor] = None
    prev_weights: torch.Tensor = torch.zeros(1, N_ASSETS, device=device)

    for epoch in range(N_EPOCHS):
        total_loss: float = 0.0
        h_state = None  # Reset hidden state at start of epoch (or keep if contiguous)
        
        # Iterate through time t
        # Note: In real implementation, batching is done randomly for transformer, 
        # but SSM requires sequential order. The paper suggests batching "all assets for a single month" [cite: 529]
        
        for t in range(T - 1):
            optimizer.zero_grad()

            # Inputs for time t
            x_t: torch.Tensor = data_X[t].unsqueeze(0).to(device)  # (1, N, D)
            r_t1: torch.Tensor = data_R[t + 1].unsqueeze(0).to(device)  # (1, N) Return at t+1
            c_t: torch.Tensor = data_C[t].unsqueeze(0).to(device)  # (1, N)

            # 1. Forward Pass
            # h_state carries memory from t-1 to t [cite: 261]
            weights_t: torch.Tensor
            h_state_new: torch.Tensor
            weights_t, h_state_new = model(x_t, h_state)
            
            # Detach hidden state to prevent backprop through entire history (Truncated BPTT)
            # Or keep graph if sequence length is manageable
            h_state = h_state_new.detach() 
            
            # 2. Calculate Drifted Weights w_{t-1}^+ [cite: 447]
            # Simple approx: w_prev (assuming small intra-month returns for drift calc)
            # Ideally: w_prev * (1 + r_t) / sum(...)
            weights_prev_plus: torch.Tensor = prev_weights  # Simplified for demo

            # 3. Compute Loss
            loss: torch.Tensor
            metrics: dict[str, float]
            loss, metrics = loss_fn(weights_t, r_t1, weights_prev_plus, c_t, cvar_nu)
            
            # 4. Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clipping [cite: 528]
            optimizer.step()
            
            # Update prev weights for next step cost calculation
            # Use .detach() so we don't backprop into t-1 in the next step via TC term
            prev_weights = weights_t.detach()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}: Avg Loss {total_loss/T:.4f} | CVaR Nu: {cvar_nu.item():.4f}")

if __name__ == "__main__":
    train_sa_sdf()

