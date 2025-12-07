import numpy as np
import torch  # Add this import

def inference_service(mu, sigma):
    """Computes pressure effects and 94% HDIs."""
    pressure_A_mean = mu[1].item()
    pressure_A_std = sigma[1].item()
    pressure_B_mean = mu[1].item() + mu[3].item()
    pressure_B_std = torch.sqrt(sigma[1]**2 + sigma[3]**2).item()  # Use torch.sqrt
    
    z = 1.88  # Approx for 94% HDI (normal dist)
    hdi_A = [pressure_A_mean - z * pressure_A_std, pressure_A_mean + z * pressure_A_std]
    hdi_B = [pressure_B_mean - z * pressure_B_std, pressure_B_mean + z * pressure_B_std]
    
    verdict = "Buy Bowler B" if pressure_B_mean > pressure_A_mean and hdi_B[0] > 0 else "Buy Bowler A"
    print(f"Pressure A: {pressure_A_mean:.4f} (HDI: {hdi_A})")
    print(f"Pressure B: {pressure_B_mean:.4f} (HDI: {hdi_B})")
    print(f"Verdict: {verdict}")
    return pressure_A_mean, pressure_A_std, pressure_B_mean, pressure_B_std, verdict