import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np 

class Policy(nn.Module):
      def __init__(self, input_dim, n_actions, embedding_dim=256):
            super(Policy, self).__init__()

            self.input_dim = input_dim
            self.embedding_dim = embedding_dim
            self.output_dim = n_actions 

            self.policy_net = nn.Sequential(
                  nn.Linear(self.input_dim, self.output_dim)
                  # nn.ReLU(),
                  # nn.Linear(self.embedding_dim, self.output_dim)
            )

      def forward(self, categorical_probs):
            action_logits = self.policy_net(categorical_probs)
            action_probs = F.softmax(action_logits, dim = -1)
            return action_probs


class SacActor(nn.Module):
      def __init__(self, state_embedding_dim, action_dim, hidden_dim):
            super(SacActor, self).__init__()

            self.LOG_STD_MIN, self.LOG_STD_MAX = -5, 2
            self.action_scale = 1.0
            self.action_bias = 0

            self.fc1 = nn.Linear(state_embedding_dim, hidden_dim)

            self.mean = nn.Linear(hidden_dim, action_dim)

            self.log_std_dev = nn.Linear(hidden_dim, action_dim)

            self.apply(self._weights_init)

      def _weights_init(self, m):
            
            if isinstance(m, nn.Linear):
                  nn.init.xavier_uniform_(m.weight)
                  nn.init.constant_(m.bias, 0)

      def forward(self, state_embedding):
            """Forward pass through the policy network."""
            x = F.relu(self.fc1(state_embedding))
            
            mean = self.mean(x)
            log_std = self.log_std_dev(x)
            log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
            
            return mean, log_std

      def sample(self, state):
            """Sample an action using the reparameterization trick."""
            mean, log_std = self.forward(state)
            std = log_std.exp()
            
            # Reparameterization trick: sample from Normal(mean, std)
            normal = Normal(mean, std)
            x_t = normal.rsample()  # differentiable sample
            y_t = torch.tanh(x_t)   # squash to (-1, 1)
            action = y_t * self.action_scale + self.action_bias
            
            # Compute log probability with tanh correction
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            
            mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
            
            return action, log_prob, mean_action

