import numpy as np 
import torch 
import torch.nn as nn 

class ValueNetwork(nn.Module):
      def __init__(self, embedding_dim, action_dim):  # Add action_dim parameter
            super(ValueNetwork, self).__init__()
            
            self.embedding_dim = embedding_dim
            self.action_dim = action_dim
            
            # Shared backbone for state embeddings
            self.shared_backbone = nn.Sequential(
                  nn.Linear(embedding_dim, 256),
                  nn.ReLU(),
                  nn.Linear(256, 256),
                  nn.ReLU()
            )
            
            # Value head
            self.value_head = nn.Linear(256, 1)
            
            # Q-value heads (two for SAC)
            self.q1_head = nn.Linear(256 + action_dim, 1)
            self.q2_head = nn.Linear(256 + action_dim, 1)
            
            self.apply(self._weights_init)

      def forward(self, state_embedding, action=None):
            shared_features = self.shared_backbone(state_embedding)
            
            # Value output
            value = self.value_head(shared_features)
            
            # Q-value outputs if action is provided
            if action is not None:
                  # Ensure action has correct shape
                  if len(action.shape) == 1:
                        action = action.unsqueeze(-1)
                  q_input = torch.cat([shared_features, action], dim=-1)
                  q1_value = self.q1_head(q_input)
                  q2_value = self.q2_head(q_input)
                  return value, q1_value, q2_value
            
            return value

      def get_q_values(self, state_embedding, action):
            """Get only Q-values without value"""
            shared_features = self.shared_backbone(state_embedding)
            # Ensure action has correct shape
            if len(action.shape) == 1:
                  action = action.unsqueeze(-1)
            q_input = torch.cat([shared_features, action], dim=-1)
            q1_value = self.q1_head(q_input)
            q2_value = self.q2_head(q_input)
            return q1_value, q2_value
      
      def _weights_init(self, m):
            if isinstance(m, nn.Linear):
                  nn.init.xavier_uniform_(m.weight)
                  nn.init.constant_(m.bias, 0)