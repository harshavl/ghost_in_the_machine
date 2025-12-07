import matplotlib.pyplot as plt
import torch
from torch.distributions import Normal
import numpy as np

def visualization_service(pressure_A_mean, pressure_A_std, pressure_B_mean, pressure_B_std):
    """Generates posterior plots."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    x_a = np.linspace(pressure_A_mean - 4*pressure_A_std, pressure_A_mean + 4*pressure_A_std, 100)
    ax[0].plot(x_a, np.exp(Normal(pressure_A_mean, pressure_A_std).log_prob(torch.tensor(x_a))).numpy())
    ax[0].set_title('Posterior: Pressure Effect A')
    
    x_b = np.linspace(pressure_B_mean - 4*pressure_B_std, pressure_B_mean + 4*pressure_B_std, 100)
    ax[1].plot(x_b, np.exp(Normal(pressure_B_mean, pressure_B_std).log_prob(torch.tensor(x_b))).numpy())
    ax[1].set_title('Posterior: Pressure Effect B')
    
    plt.savefig('../docs/posteriors.png')  # Correct relative path from notebooks/
    print("Posteriors plotted and saved as '../docs/posteriors.png'.")