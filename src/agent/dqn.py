"""
Architecture Dueling Double Deep Q-Network (D3QN) professionnelle.
- Dueling Architecture : sépare la valeur de l'état V(s) et l'avantage de l'action A(s, a).
- Double DQN : prévient la surestimation des Q-values.
- Exportable pour déploiement temps réel.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agent.buffer import PrioritizedReplayBuffer

class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int = 10, action_dim: int = 9, hidden_dim: int = 128):
        super(DuelingQNetwork, self).__init__()
        
        # Tronçon commun d'extraction d'attributs
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Branche Valeur d'état V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Branche Avantage de l'action A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values

class D3QNAgent:
    def __init__(
        self,
        state_dim: int = 10,
        action_dim: int = 9,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,  # Soft update polyak
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.03,
        epsilon_decay: float = 0.996,
        buffer_capacity: int = 50_000,
        batch_size: int = 64,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = torch.device(device)

        # Réseaux Online et Target
        self.online_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber Loss

        self.buffer = PrioritizedReplayBuffer(state_dim, capacity=buffer_capacity)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        if not evaluate and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_dim))

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return int(q_values.argmax(dim=-1).item())

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size, self.device)

        # Q_pred = Q_online(s, a)
        q_values = self.online_net(states)
        q_pred = q_values.gather(1, actions)

        # Double DQN : l'action est choisie par le réseau online, évaluée par le target network
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=-1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            q_target = rewards + (1.0 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        # Soft update du target network (Polyak averaging) : target = tau * online + (1 - tau) * target
        for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        torch.save({
            "model_state": self.online_net.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "epsilon": self.epsilon
        }, filepath)

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        self.online_net.load_state_dict(checkpoint["model_state"])
        self.target_net.load_state_dict(checkpoint["model_state"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)

