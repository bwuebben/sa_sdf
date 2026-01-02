from typing import Optional, List, Dict, Tuple
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from sa_sdf.model import SASDF
from sa_sdf.loss import SASDFLoss


def calculate_sharpe_ratio(returns: np.ndarray, annualization: float = 12.0) -> float:
    """Calculate annualized Sharpe ratio."""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    return np.sqrt(annualization) * np.mean(returns) / np.std(returns)


def calculate_max_drawdown(cumulative_returns: np.ndarray) -> float:
    """Calculate maximum drawdown from cumulative returns."""
    cum_ret = 1.0 + cumulative_returns
    running_max = np.maximum.accumulate(cum_ret)
    drawdown = (cum_ret - running_max) / running_max
    return np.min(drawdown) * 100  # Return as percentage


def print_separator(char: str = "=", length: int = 80) -> None:
    """Print a separator line."""
    print(char * length)


def create_output_directory() -> str:
    """Create timestamped output directory for this run."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("output", f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    return output_dir


def plot_cumulative_returns(returns: np.ndarray, output_dir: str) -> None:
    """Plot cumulative returns over time."""
    cum_returns = np.cumsum(returns) * 100
    months = np.arange(len(returns))

    plt.figure(figsize=(12, 6))
    plt.plot(months, cum_returns, linewidth=2, color='#2E86AB')
    plt.fill_between(months, 0, cum_returns, alpha=0.3, color='#2E86AB')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Cumulative Return (%)', fontsize=12)
    plt.title('SA-SDF Portfolio Cumulative Returns', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cumulative_returns.png'), dpi=300)
    plt.close()


def plot_drawdown(returns: np.ndarray, output_dir: str) -> None:
    """Plot drawdown over time."""
    cum_returns = np.cumsum(returns)
    cum_ret = 1.0 + cum_returns
    running_max = np.maximum.accumulate(cum_ret)
    drawdown = (cum_ret - running_max) / running_max * 100
    months = np.arange(len(returns))

    plt.figure(figsize=(12, 6))
    plt.fill_between(months, 0, drawdown, alpha=0.5, color='#A4243B')
    plt.plot(months, drawdown, linewidth=2, color='#A4243B')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Drawdown (%)', fontsize=12)
    plt.title('Portfolio Drawdown Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'drawdown.png'), dpi=300)
    plt.close()


def plot_loss_decomposition(epoch_history: List[Dict], output_dir: str) -> None:
    """Plot loss components over epochs."""
    epochs = np.arange(1, len(epoch_history) + 1)
    total_loss = [e['avg_loss'] for e in epoch_history]
    pricing_error = [e['avg_pricing'] for e in epoch_history]
    tc_cost = [e['avg_tc'] for e in epoch_history]
    cvar = [e['avg_cvar'] for e in epoch_history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Total Loss
    axes[0, 0].plot(epochs, total_loss, marker='o', linewidth=2, color='#2E86AB')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Pricing Error
    axes[0, 1].plot(epochs, pricing_error, marker='o', linewidth=2, color='#F77F00')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Pricing Error')
    axes[0, 1].set_title('Pricing Error Component', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Transaction Costs
    axes[1, 0].plot(epochs, tc_cost, marker='o', linewidth=2, color='#06A77D')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('TC Cost')
    axes[1, 0].set_title('Transaction Cost Component', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # CVaR
    axes[1, 1].plot(epochs, cvar, marker='o', linewidth=2, color='#A4243B')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('CVaR')
    axes[1, 1].set_title('CVaR Penalty Component', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_decomposition.png'), dpi=300)
    plt.close()


def plot_performance_metrics(epoch_history: List[Dict], output_dir: str) -> None:
    """Plot Sharpe ratio, volatility, and max drawdown evolution."""
    epochs = np.arange(1, len(epoch_history) + 1)
    sharpe = [e['sharpe'] for e in epoch_history]
    volatility = [e['volatility'] for e in epoch_history]
    max_dd = [abs(e['max_dd']) for e in epoch_history]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Sharpe Ratio
    axes[0].plot(epochs, sharpe, marker='o', linewidth=2, color='#2E86AB')
    axes[0].axhline(y=1.43, color='green', linestyle='--', alpha=0.5, label='Paper Target (1.43)')
    axes[0].axhline(y=1.15, color='red', linestyle='--', alpha=0.5, label='KKMU Baseline (1.15)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Sharpe Ratio')
    axes[0].set_title('Sharpe Ratio Evolution', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Volatility
    axes[1].plot(epochs, volatility, marker='o', linewidth=2, color='#F77F00')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Annualized Volatility (%)')
    axes[1].set_title('Volatility Evolution', fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Max Drawdown
    axes[2].plot(epochs, max_dd, marker='o', linewidth=2, color='#A4243B')
    axes[2].axhline(y=26.8, color='green', linestyle='--', alpha=0.5, label='Paper Target (26.8%)')
    axes[2].axhline(y=33.4, color='red', linestyle='--', alpha=0.5, label='KKMU Baseline (33.4%)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Max Drawdown (%)')
    axes[2].set_title('Maximum Drawdown Evolution', fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_metrics.png'), dpi=300)
    plt.close()


def plot_turnover(turnovers: np.ndarray, output_dir: str) -> None:
    """Plot turnover over time."""
    months = np.arange(len(turnovers))
    avg_turnover = np.mean(turnovers) * 100

    plt.figure(figsize=(12, 6))
    plt.plot(months, turnovers * 100, linewidth=1.5, alpha=0.7, color='#06A77D')
    plt.axhline(y=avg_turnover, color='#2E86AB', linestyle='--', linewidth=2,
                label=f'Average: {avg_turnover:.1f}%')
    plt.axhline(y=38.6, color='green', linestyle='--', alpha=0.5, label='Paper Target (38.6%)')
    plt.axhline(y=71.2, color='red', linestyle='--', alpha=0.5, label='KKMU Baseline (71.2%)')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Turnover (%)', fontsize=12)
    plt.title('Monthly Portfolio Turnover', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'turnover.png'), dpi=300)
    plt.close()


def plot_portfolio_composition(long_positions: np.ndarray, short_positions: np.ndarray,
                               n_assets: int, output_dir: str) -> None:
    """Plot long/short position counts over time."""
    months = np.arange(len(long_positions))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Position counts
    axes[0].plot(months, long_positions, linewidth=2, color='#06A77D', label='Long Positions')
    axes[0].plot(months, short_positions, linewidth=2, color='#A4243B', label='Short Positions')
    axes[0].axhline(y=n_assets/2, color='black', linestyle='--', alpha=0.3, label='Market Neutral')
    axes[0].set_xlabel('Month')
    axes[0].set_ylabel('Number of Positions')
    axes[0].set_title('Portfolio Composition: Long vs Short', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Net exposure
    net_exposure = (long_positions - short_positions) / n_assets * 100
    axes[1].plot(months, net_exposure, linewidth=2, color='#2E86AB')
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[1].fill_between(months, 0, net_exposure, alpha=0.3, color='#2E86AB')
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel('Net Exposure (%)')
    axes[1].set_title('Net Long/Short Exposure', fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'portfolio_composition.png'), dpi=300)
    plt.close()


def plot_cvar_evolution(cvar_history: List[float], output_dir: str) -> None:
    """Plot CVaR threshold (nu) evolution."""
    epochs = np.arange(1, len(cvar_history) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, cvar_history, marker='o', linewidth=2, color='#A4243B')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('CVaR Threshold (ν)', fontsize=12)
    plt.title('CVaR Threshold Evolution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cvar_evolution.png'), dpi=300)
    plt.close()


def plot_portfolio_weights_heatmap(weights_history: np.ndarray, output_dir: str) -> None:
    """Plot heatmap of portfolio weights over time (sample)."""
    # Sample every 10th timestep to keep visualization manageable
    sampled_weights = weights_history[::10, :20]  # First 20 assets

    plt.figure(figsize=(14, 6))
    plt.imshow(sampled_weights.T, aspect='auto', cmap='RdBu_r', vmin=-0.1, vmax=0.1)
    plt.colorbar(label='Weight')
    plt.xlabel('Time (sampled every 10 months)', fontsize=12)
    plt.ylabel('Asset Index', fontsize=12)
    plt.title('Portfolio Weights Heatmap (First 20 Assets)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'weights_heatmap.png'), dpi=300)
    plt.close()


def generate_report(output_dir: str, hyperparams: Dict, final_metrics: Dict,
                   epoch_history: List[Dict], training_time: float) -> None:
    """Generate comprehensive markdown report."""
    report_path = os.path.join(output_dir, 'report.md')

    with open(report_path, 'w') as f:
        f.write("# SA-SDF Training Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # Hyperparameters
        f.write("## Hyperparameters\n\n")
        f.write(f"- **Epochs:** {hyperparams['n_epochs']}\n")
        f.write(f"- **Learning Rate:** {hyperparams['lr']}\n")
        f.write(f"- **Training Months:** {hyperparams['T']}\n")
        f.write(f"- **Assets:** {hyperparams['n_assets']}\n")
        f.write(f"- **Characteristics:** {hyperparams['n_chars']}\n")
        f.write(f"- **State Tokens:** {hyperparams['state_tokens']}\n")
        f.write(f"- **Device:** {hyperparams['device']}\n")
        f.write(f"- **λ_TC:** {hyperparams['lambda_tc']}\n")
        f.write(f"- **λ_CVaR:** {hyperparams['lambda_cvar']}\n")
        f.write(f"- **Training Time:** {training_time:.2f} seconds\n\n")

        # Final Performance Metrics
        f.write("## Final Performance Metrics\n\n")
        f.write("### Portfolio Performance\n")
        f.write(f"- **Net Sharpe Ratio:** {final_metrics['sharpe']:.3f}\n")
        f.write(f"- **Total Return:** {final_metrics['total_return']:+.2f}%\n")
        f.write(f"- **Annualized Volatility:** {final_metrics['volatility']:.2f}%\n")
        f.write(f"- **Maximum Drawdown:** {final_metrics['max_drawdown']:.2f}%\n")
        f.write(f"- **Average Monthly Return:** {final_metrics['avg_monthly_return']:+.2f}%\n\n")

        f.write("### Transaction Costs\n")
        f.write(f"- **Average Monthly Turnover:** {final_metrics['avg_turnover']:.2f}%\n\n")

        # Paper Comparison
        f.write("## Paper Benchmark Comparison\n\n")
        f.write("| Metric | SA-SDF (Paper) | KKMU Baseline | This Run |\n")
        f.write("|--------|----------------|---------------|----------|\n")
        f.write(f"| Net Sharpe Ratio | 1.43 | 1.15 | {final_metrics['sharpe']:.3f} |\n")
        f.write(f"| Max Drawdown | 26.8% | 33.4% | {abs(final_metrics['max_drawdown']):.1f}% |\n")
        f.write(f"| Monthly Turnover | 38.6% | 71.2% | {final_metrics['avg_turnover']:.1f}% |\n\n")

        # Epoch-by-Epoch Results
        f.write("## Training Progress\n\n")
        f.write("| Epoch | Loss | Sharpe | Volatility | Max DD | Turnover |\n")
        f.write("|-------|------|--------|------------|--------|----------|\n")
        for i, epoch in enumerate(epoch_history, 1):
            f.write(f"| {i} | {epoch['avg_loss']:.4f} | {epoch['sharpe']:.3f} | "
                   f"{epoch['volatility']:.2f}% | {abs(epoch['max_dd']):.2f}% | "
                   f"{epoch['avg_turnover']:.2f}% |\n")

        f.write("\n---\n\n")
        f.write("## Visualizations\n\n")
        f.write("- `cumulative_returns.png` - Cumulative portfolio returns\n")
        f.write("- `drawdown.png` - Drawdown over time\n")
        f.write("- `loss_decomposition.png` - Loss components by epoch\n")
        f.write("- `performance_metrics.png` - Sharpe, volatility, and drawdown evolution\n")
        f.write("- `turnover.png` - Monthly turnover\n")
        f.write("- `portfolio_composition.png` - Long/short positions\n")
        f.write("- `cvar_evolution.png` - CVaR threshold evolution\n")
        f.write("- `weights_heatmap.png` - Portfolio weights heatmap\n\n")

        f.write("---\n\n")
        f.write("## Key Findings\n\n")

        # Analysis
        sharpe_vs_paper = (final_metrics['sharpe'] / 1.43 - 1) * 100
        dd_vs_paper = (abs(final_metrics['max_drawdown']) / 26.8 - 1) * 100
        turnover_vs_paper = (final_metrics['avg_turnover'] / 38.6 - 1) * 100

        f.write(f"- Sharpe ratio is {sharpe_vs_paper:+.1f}% vs paper target\n")
        f.write(f"- Max drawdown is {dd_vs_paper:+.1f}% vs paper target\n")
        f.write(f"- Turnover is {turnover_vs_paper:+.1f}% vs paper target\n\n")

        f.write("**Note:** This training uses mock/synthetic data. Real-world performance requires:\n")
        f.write("- Historical stock characteristics (132 dimensions from Jensen et al. 2023)\n")
        f.write("- CRSP return data\n")
        f.write("- Transaction cost estimates\n")
        f.write("- Proper out-of-sample backtesting with rolling windows\n")

    print(f"📄 Report saved: {report_path}")


def train_sa_sdf() -> None:
    import time
    start_time = time.time()

    # --- Hyperparameters [cite: 573] ---
    BATCH_SIZE: int = 1
    SEQ_LEN: int = 60
    N_EPOCHS: int = 5
    LR: float = 5e-5
    N_ASSETS: int = 50
    N_CHARS: int = 132
    STATE_TOKENS: int = 8
    T: int = 120

    # Create output directory
    output_dir = create_output_directory()

    # Mock Data Generation
    data_X: torch.Tensor = torch.rand(T, N_ASSETS, N_CHARS) - 0.5
    data_R: torch.Tensor = torch.randn(T, N_ASSETS) * 0.05
    data_C: torch.Tensor = torch.rand(T, N_ASSETS) * 0.001

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_separator()
    print("SA-SDF PORTFOLIO OPTIMIZATION TRAINING")
    print_separator()
    print(f"Device: {device}")
    print(f"Training Data: {T} months, {N_ASSETS} assets, {N_CHARS} characteristics")
    print(f"Epochs: {N_EPOCHS} | Learning Rate: {LR} | State Tokens: {STATE_TOKENS}")
    print_separator()

    # Initialize Model
    model: SASDF = SASDF(num_chars=N_CHARS, num_state_tokens=STATE_TOKENS).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params:,} (trainable: {trainable_params:,})")

    # Initialize Loss
    cvar_nu: torch.Tensor = torch.tensor(0.0, requires_grad=True, device=device)
    loss_fn: SASDFLoss = SASDFLoss(lambda_tc=0.01, lambda_cvar=0.5)
    print(f"Loss Components: λ_TC={loss_fn.lambda_tc}, λ_CVaR={loss_fn.lambda_cvar}, τ={loss_fn.tau}")
    print_separator()

    # Optimizer
    optimizer: optim.AdamW = optim.AdamW(list(model.parameters()) + [cvar_nu], lr=LR, weight_decay=1e-3)

    print("\nStarting Training...\n")

    # Training Loop
    model.train()
    epoch_history: List[Dict] = []
    cvar_history: List[float] = []
    all_weights_history: List[np.ndarray] = []

    for epoch in range(N_EPOCHS):
        h_state: Optional[torch.Tensor] = None
        prev_weights: torch.Tensor = torch.zeros(1, N_ASSETS, device=device)

        epoch_metrics: dict = {
            'total_loss': 0.0,
            'pricing_error': 0.0,
            'tc_cost': 0.0,
            'cvar': 0.0,
            'returns': [],
            'turnovers': [],
            'num_long': [],
            'num_short': [],
            'concentration': []
        }

        pbar = tqdm(range(T - 1), desc=f"Epoch {epoch+1}/{N_EPOCHS}", ncols=100)

        for t in pbar:
            optimizer.zero_grad()

            x_t: torch.Tensor = data_X[t].unsqueeze(0).to(device)
            r_t1: torch.Tensor = data_R[t + 1].unsqueeze(0).to(device)
            c_t: torch.Tensor = data_C[t].unsqueeze(0).to(device)

            weights_t: torch.Tensor
            h_state_new: torch.Tensor
            weights_t, h_state_new = model(x_t, h_state)
            h_state = h_state_new.detach()

            weights_prev_plus: torch.Tensor = prev_weights

            loss: torch.Tensor
            metrics: dict[str, float]
            loss, metrics = loss_fn(weights_t, r_t1, weights_prev_plus, c_t, cvar_nu)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Track metrics
            epoch_metrics['total_loss'] += loss.item()
            epoch_metrics['pricing_error'] += metrics['price_error']
            epoch_metrics['tc_cost'] += metrics['tc_cost']
            epoch_metrics['cvar'] += metrics['cvar']
            epoch_metrics['returns'].append(metrics['avg_ret'])

            w_np = weights_t.detach().cpu().numpy()[0]
            epoch_metrics['num_long'].append(np.sum(w_np > 0.01))
            epoch_metrics['num_short'].append(np.sum(w_np < -0.01))
            epoch_metrics['concentration'].append(np.sum(np.abs(w_np[:5])))

            turnover = torch.abs(weights_t - prev_weights).sum().item()
            epoch_metrics['turnovers'].append(turnover)

            prev_weights = weights_t.detach()

            # Save weights for last epoch
            if epoch == N_EPOCHS - 1:
                all_weights_history.append(w_np)

            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Ret': f"{metrics['avg_ret']*100:.2f}%"
            })

        # Calculate epoch statistics
        returns_array = np.array(epoch_metrics['returns'])
        cum_returns = np.cumsum(returns_array)

        epoch_summary = {
            'avg_loss': epoch_metrics['total_loss'] / (T - 1),
            'avg_pricing': epoch_metrics['pricing_error'] / (T - 1),
            'avg_tc': epoch_metrics['tc_cost'] / (T - 1),
            'avg_cvar': epoch_metrics['cvar'] / (T - 1),
            'avg_return': np.mean(returns_array) * 100,
            'total_return': (1 + cum_returns[-1]) * 100 - 100,
            'volatility': np.std(returns_array) * np.sqrt(12) * 100,
            'sharpe': calculate_sharpe_ratio(returns_array),
            'max_dd': calculate_max_drawdown(cum_returns),
            'avg_turnover': np.mean(epoch_metrics['turnovers']) * 100,
            'avg_long': np.mean(epoch_metrics['num_long']),
            'avg_short': np.mean(epoch_metrics['num_short']),
            'avg_concentration': np.mean(epoch_metrics['concentration']) * 100,
            'returns_series': returns_array,
            'turnovers_series': np.array(epoch_metrics['turnovers']),
            'long_series': np.array(epoch_metrics['num_long']),
            'short_series': np.array(epoch_metrics['num_short'])
        }

        epoch_history.append(epoch_summary)
        cvar_history.append(cvar_nu.item())

        # Print detailed epoch summary
        print(f"\n{'─' * 80}")
        print(f"EPOCH {epoch+1} SUMMARY")
        print(f"{'─' * 80}")
        print(f"\n📊 Loss Decomposition:")
        print(f"  Total Loss:      {epoch_summary['avg_loss']:.4f}")
        print(f"  Pricing Error:   {epoch_summary['avg_pricing']:.4f}")
        print(f"  Trans. Costs:    {epoch_summary['avg_tc']:.6f}")
        print(f"  CVaR Penalty:    {epoch_summary['avg_cvar']:.4f}")
        print(f"  CVaR Nu (VaR):   {cvar_nu.item():.4f}")

        print(f"\n💰 Portfolio Performance:")
        print(f"  Avg Monthly Return:  {epoch_summary['avg_return']:+.2f}%")
        print(f"  Total Return:        {epoch_summary['total_return']:+.2f}%")
        print(f"  Annualized Vol:      {epoch_summary['volatility']:.2f}%")
        print(f"  Sharpe Ratio:        {epoch_summary['sharpe']:.3f}")
        print(f"  Max Drawdown:        {epoch_summary['max_dd']:.2f}%")

        print(f"\n🔄 Transaction Costs:")
        print(f"  Avg Monthly Turnover: {epoch_summary['avg_turnover']:.2f}%")
        print(f"  Total TC Impact:      {epoch_summary['avg_tc']*100:.3f}%")

        print(f"\n📈 Portfolio Composition:")
        print(f"  Avg Long Positions:   {epoch_summary['avg_long']:.1f} / {N_ASSETS}")
        print(f"  Avg Short Positions:  {epoch_summary['avg_short']:.1f} / {N_ASSETS}")
        print(f"  Top-5 Concentration:  {epoch_summary['avg_concentration']:.1f}%")

        print()

    # Final evaluation
    print_separator("=")
    print("GENERATING VISUALIZATIONS AND REPORT...")
    print_separator("=")

    model.eval()
    with torch.no_grad():
        h_state = None
        prev_weights = torch.zeros(1, N_ASSETS, device=device)
        final_returns: List[float] = []
        final_turnovers: List[float] = []

        for t in range(T - 1):
            x_t = data_X[t].unsqueeze(0).to(device)
            r_t1 = data_R[t + 1].unsqueeze(0).to(device)

            weights_t, h_state = model(x_t, h_state)

            port_ret = (weights_t * r_t1).sum().item()
            turnover = torch.abs(weights_t - prev_weights).sum().item()

            final_returns.append(port_ret)
            final_turnovers.append(turnover)
            prev_weights = weights_t

    final_returns_array = np.array(final_returns)
    final_cum_returns = np.cumsum(final_returns_array)

    final_metrics = {
        'sharpe': calculate_sharpe_ratio(final_returns_array),
        'total_return': (1 + final_cum_returns[-1]) * 100 - 100,
        'volatility': np.std(final_returns_array) * np.sqrt(12) * 100,
        'max_drawdown': calculate_max_drawdown(final_cum_returns),
        'avg_monthly_return': np.mean(final_returns_array) * 100,
        'avg_turnover': np.mean(final_turnovers) * 100
    }

    # Generate all plots
    print("📊 Generating charts...")
    plot_cumulative_returns(final_returns_array, output_dir)
    plot_drawdown(final_returns_array, output_dir)
    plot_loss_decomposition(epoch_history, output_dir)
    plot_performance_metrics(epoch_history, output_dir)
    plot_turnover(np.array(final_turnovers), output_dir)
    plot_portfolio_composition(
        epoch_history[-1]['long_series'],
        epoch_history[-1]['short_series'],
        N_ASSETS,
        output_dir
    )
    plot_cvar_evolution(cvar_history, output_dir)
    if all_weights_history:
        plot_portfolio_weights_heatmap(np.array(all_weights_history), output_dir)

    print("✅ All charts generated!")

    # Generate report
    training_time = time.time() - start_time
    hyperparams = {
        'n_epochs': N_EPOCHS,
        'lr': LR,
        'T': T,
        'n_assets': N_ASSETS,
        'n_chars': N_CHARS,
        'state_tokens': STATE_TOKENS,
        'device': str(device),
        'lambda_tc': loss_fn.lambda_tc,
        'lambda_cvar': loss_fn.lambda_cvar
    }

    generate_report(output_dir, hyperparams, final_metrics, epoch_history, training_time)

    print_separator("=")
    print(f"🎯 Final Net Sharpe Ratio: {final_metrics['sharpe']:.3f}")
    print(f"📁 All outputs saved to: {output_dir}")
    print_separator("=")
    print("✅ Training session completed successfully!\n")


if __name__ == "__main__":
    train_sa_sdf()
