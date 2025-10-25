from collections import deque, namedtuple
import random
import torch
import numpy as np

class Buffer:
      def __init__(self, buffer_length, frame_stack=3) -> None:
            self.buffer_length = buffer_length
            self.buffer = deque(maxlen=self.buffer_length)
            self.experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])
            self.frame_stack = frame_stack

      def sample(self, batch_size):
            exp = random.sample(self.buffer, k=batch_size)
            states, actions, rewards, next_states, dones = zip(*exp)
            
            # Stack states and next_states - ensure proper shape
            states = torch.stack([s for s in states])
            actions = torch.stack([a for a in actions])
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.stack([ns for ns in next_states])
            dones = torch.tensor(dones, dtype=torch.float32)
            
            return states, actions, rewards, next_states, dones

      def add(self, state, action, reward, next_state, done):
            # Convert to tensors if they're numpy arrays
            if isinstance(state, np.ndarray):
                  state = torch.FloatTensor(state.copy())
            if isinstance(next_state, np.ndarray):
                  next_state = torch.FloatTensor(next_state.copy())
            if isinstance(action, np.ndarray):
                  action = torch.FloatTensor(action.copy())
                  
            exp = self.experience(state, action, reward, next_state, done)
            self.buffer.append(exp)

      def __len__(self):
            return len(self.buffer)