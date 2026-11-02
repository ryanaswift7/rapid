import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def main():
    # --- Setup & Styling ---
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 14,           # Increased base font size
        "axes.titlesize": 16,      # Added and increased title size
        "axes.labelsize": 16,      # Increased axis label size
        "legend.fontsize": 14,     # Increased legend text size
        "xtick.labelsize": 14,     # Increased tick mark label size
        "ytick.labelsize": 14      # Increased tick mark label size
    })

    x = np.linspace(-9, 11, 1000)

    # --- Define Target Distribution P (Drastic Trimodal) ---
    # Weights: 10% left, 70% center, 20% right
    p_x = 0.1 * norm.pdf(x, -4, 0.8) + 0.7 * norm.pdf(x, 1, 1.2) + 0.2 * norm.pdf(x, 6, 1.0)

    # --- Define Approximations Q ---
    # 1. Forward KL: Analytically matches the exact mean and variance of P
    mean_p = 0.1 * (-4) + 0.7 * (1) + 0.2 * (6)
    var_p = (0.1 * (0.8**2 + (-4 - mean_p)**2) + 
            0.7 * (1.2**2 + (1 - mean_p)**2) + 
            0.2 * (1.0**2 + (6 - mean_p)**2))
    std_p = np.sqrt(var_p)

    q_fwd = norm.pdf(x, mean_p, std_p)

    # 2. Reverse KL: Locks onto the dominant mode
    q_rev = norm.pdf(x, 1.0, 1.2)

    # 3. Jensen-Shannon: Biased towards dominant mass but compromises
    q_js = norm.pdf(x, 1.2, 2.0)

    # --- Plotting ---
    # Increased width from 8 to 10 to make it wider
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)

    # Target distribution
    ax.fill_between(x, p_x, color='gray', alpha=0.2, label='Target $P(x)$ (Trimodal GMM)')
    ax.plot(x, p_x, color='black', lw=1.5, linestyle='--')

    # Divergence Approximations
    ax.plot(x, q_fwd, label='Forward KL (Mean-seeking)', color='#1f77b4', lw=2.5)
    ax.plot(x, q_rev, label='Reverse KL (Mode-seeking)', color='#d62728', lw=2.5)
    ax.plot(x, q_js, label='Jensen-Shannon', color='#2ca02c', lw=2.5)

    # Clean up axes
    # ax.set_title("Behavior of Divergence Measures", pad=15)
    ax.set_xlabel("State Space ($x$)")
    ax.set_ylabel("Density")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc='upper right')

    plt.tight_layout()

    # Save directly as a PDF for the LaTeX document
    plt.savefig('divergences_trimodal.pdf', format='pdf', bbox_inches='tight')
if __name__ == "__main__":
    main()