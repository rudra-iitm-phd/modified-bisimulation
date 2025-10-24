import numpy as np 
import torch 
import torch.nn as nn 


class ValueNetwork(nn.Module):
      def __init__(self, embedding_dim):
            super(ValueNetwork, self).__init__()

            self.input_dim = embedding_dim

            self.value = nn.Sequential(
                  nn.Linear(self.input_dim, 1)
            )

            self.apply(self._weights_init)

      def forward(self, state_embedding):

            return self.value(state_embedding)

      def _weights_init(self, m):
            
            if isinstance(m, nn.Linear):
                  nn.init.xavier_uniform_(m.weight)
                  nn.init.constant_(m.bias, 0)