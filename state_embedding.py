import numpy as np 
import torch
import torch.nn as nn 


class StateEmbedding(nn.Module):
      def __init__(self, state_dim, output_dim):
            super(StateEmbedding,self).__init__()

            """
            Takes a raw state and makes a representation

            """

            self.input_dim = state_dim
            self.output_dim = output_dim

            self.phi = nn.Sequential(
                  nn.Linear(in_features= self.input_dim, out_features = self.output_dim),
                  nn.ReLU(),
                  nn.Linear(self.output_dim, out_features = self.output_dim),
                  nn.ReLU()
            )

      def forward(self, state):
            return self.phi(state)


            


