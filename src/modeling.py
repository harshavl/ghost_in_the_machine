import torch
from torch.distributions import Normal, Bernoulli
import torch.optim as optim

def modeling_service(X, y, num_iters=10000, mc_samples=50, lr=0.005):
    """Trains Bayesian logistic regression using VI."""
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    n, num_params = X_t.shape  # Renamed p to num_params
    mu = torch.zeros(num_params, requires_grad=True)
    log_sigma = torch.full((num_params,), -2.0, requires_grad=True)
    optimizer = optim.Adam([mu, log_sigma], lr=lr)
    
    for i in range(num_iters):
        optimizer.zero_grad()
        sigma = torch.exp(log_sigma)
        eps = torch.randn((mc_samples, num_params))
        beta_samples = mu + sigma * eps
        logit = torch.matmul(X_t, beta_samples.t())
        probs = torch.sigmoid(logit)  # Renamed p to probs
        log_lik = Bernoulli(probs=probs).log_prob(y_t.unsqueeze(1)).sum(dim=0)
        prior = Normal(0, 10.0)
        log_prior = prior.log_prob(beta_samples).sum(dim=1)
        q = Normal(mu, sigma)
        log_q = q.log_prob(beta_samples).sum(dim=1)
        elbo = (log_lik + log_prior - log_q).mean()
        loss = -elbo
        loss.backward()
        optimizer.step()
        if i % 2000 == 0:
            print(f'Iter {i}, Loss: {loss.item():.4f}')
    
    sigma = torch.exp(log_sigma)
    print("Model trained.")
    return mu, sigma