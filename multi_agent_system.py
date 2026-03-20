
import torch
import torch.optim as optim

from biocentric_graphcnn import BiocentricGraphCNN
from spiking_module import LIFNeuron
from iit_phi import compute_phi

NUM_AGENTS = 3
NODES = 10
DIM = 16
STEPS = 50

agents = [BiocentricGraphCNN(DIM) for _ in range(NUM_AGENTS)]
optimizers = [optim.Adam(agent.parameters(), lr=0.001) for agent in agents]

lif = LIFNeuron()

adj = torch.eye(NODES)

states = [torch.randn(NODES, DIM) for _ in range(NUM_AGENTS)]
membranes = [torch.zeros(NODES, DIM) for _ in range(NUM_AGENTS)]

for step in range(STEPS):
    realities = []

    # 🔥 DETACH STATES (CRITICAL FIX)
    states = [s.detach() for s in states]
    membranes = [m.detach() for m in membranes]

    for i in range(NUM_AGENTS):
        h = states[i]

        phi = compute_phi(h)

        spike, membranes[i] = lif(h, membranes[i])
        h = h + spike

        h, reality, action = agents[i](h, adj, phi)

        states[i] = h
        realities.append(reality)

    # Shared reality (no gradient sharing)
    shared_reality = torch.mean(torch.stack(realities), dim=0).detach()

    # Zero grads
    for opt in optimizers:
        opt.zero_grad()

    total_loss = 0

    for i in range(NUM_AGENTS):
        loss = torch.mean((realities[i] - shared_reality) ** 2)
        total_loss += loss

    # Single backward pass
    total_loss.backward()

    # Step optimizers
    for opt in optimizers:
        opt.step()

    if step % 10 == 0:
        print(f"Step {step} | Reality Mean: {shared_reality.mean().item():.4f}")
