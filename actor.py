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
      def __init__(self, state_embedding_dim, action_dim, hidden_dim=256):
            super(SacActor, self).__init__()

            self.LOG_STD_MIN, self.LOG_STD_MAX = -5, 2
            self.action_scale = 1.0
            self.action_bias = 0.0

            # Policy network
            self.net = nn.Sequential(
                  nn.Linear(state_embedding_dim, hidden_dim),
                  nn.ReLU(),
                  nn.Linear(hidden_dim, hidden_dim),
                  nn.ReLU()
            )
            
            self.mean_layer = nn.Linear(hidden_dim, action_dim)
            self.log_std_layer = nn.Linear(hidden_dim, action_dim)

            self.apply(self._weights_init)

      def _weights_init(self, m):
            if isinstance(m, nn.Linear):
                  nn.init.xavier_uniform_(m.weight)
                  nn.init.constant_(m.bias, 0)

      def forward(self, state_embedding):
            """Forward pass through the policy network."""
            features = self.net(state_embedding)
            
            mean = self.mean_layer(features)
            log_std = self.log_std_layer(features)
            log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
            
            return mean, log_std

      def sample(self, state_embedding):
            """Sample an action using the reparameterization trick."""
            mean, log_std = self.forward(state_embedding)
            std = log_std.exp()
            
            # Reparameterization trick
            normal = Normal(mean, std)
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            
            # Log probability with tanh correction
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            
            mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
            
            return action, log_prob, mean_action

      def get_action(self, state_embedding, deterministic=False):
            """Get action without log prob for inference."""
            if deterministic:
                  mean, _ = self.forward(state_embedding)
                  action = torch.tanh(mean) * self.action_scale + self.action_bias
                  return action
            else:
                  action, _, _ = self.sample(state_embedding)
                  return action
