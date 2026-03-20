
import torch

# Default Mode Network (baseline activity)
def default_mode_network(h):
    return torch.mean(h, dim=0)

# Free Will = selection among competing activations
def free_will(action_logits):
    probs = torch.softmax(action_logits, dim=-1)
    choice = torch.argmax(probs, dim=-1)
    return choice, probs

# Integrated Information Φ with DMN influence
def compute_phi(h):
    covariance = torch.cov(h.T)
    base_phi = torch.trace(covariance) / (torch.sum(covariance) + 1e-6)

    dmn = default_mode_network(h)
    dmn_strength = torch.norm(dmn)

    phi = torch.sigmoid(base_phi + 0.1 * dmn_strength)
    return phi
