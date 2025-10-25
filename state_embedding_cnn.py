import numpy as np 
import torch
import torch.nn as nn 

class StateEmbedding(nn.Module):
      def __init__(self, state_shape, output_dim, frame_stack=3):
            super(StateEmbedding, self).__init__()
            """
            Takes stacked frames (pixel observations from DMControl) and creates representation
            state_shape: (frame_stack * 3, height, width) for stacked RGB frames
            """
            self.frame_stack = frame_stack
            self.input_channels, self.height, self.width = state_shape
            self.output_dim = output_dim

            # CNN backbone for stacked frames
            self.cnn_backbone = nn.Sequential(
                  nn.Conv2d(self.input_channels, 32, kernel_size=8, stride=4),
                  nn.ReLU(),
                  nn.Conv2d(32, 64, kernel_size=4, stride=2),
                  nn.ReLU(),
                  nn.Conv2d(64, 64, kernel_size=3, stride=1),
                  nn.ReLU(),
                  nn.Flatten()
            )
            
            # Compute CNN output size dynamically
            with torch.no_grad():
                  dummy_input = torch.zeros(1, self.input_channels, self.height, self.width)
                  cnn_output = self.cnn_backbone(dummy_input)
                  cnn_output_dim = cnn_output.shape[1]
            
            print(f"CNN output dimension: {cnn_output_dim}")
            
            # MLP for final embedding
            self.mlp = nn.Sequential(
                  nn.Linear(cnn_output_dim, 512),
                  nn.ReLU(),
                  nn.Linear(512, self.output_dim)
            )

      def forward(self, state):
            # state is expected to be [batch, frame_stack * 3, height, width]
            # Ensure input is in correct format [B, C, H, W]
            if len(state.shape) == 3:
                  state = state.unsqueeze(0)  # Add batch dimension if missing
                  
            # Normalize to [0, 1] if needed
            if state.max() > 1.0:
                  state = state / 255.0
                  
            features = self.cnn_backbone(state)
            return self.mlp(features)