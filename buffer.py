from collections import deque, namedtuple
import random
import torch

class Buffer:
      def __init__(self, buffer_length) -> None:
            self.buffer_length = buffer_length
            self.buffer = deque(maxlen=self.buffer_length)
            self.experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])

      def sample(self, batch_size):
            exp = random.sample(self.buffer, k = batch_size)
            states, actions, rewards, next_states, dones = zip(*exp)
            states  = torch.stack(states)
            actions = torch.stack(actions)
            rewards = torch.tensor(rewards, dtype = torch.float32)
            next_states = torch.stack(next_states)
            dones = torch.tensor(dones, dtype = torch.float32)
            return states, actions, rewards, next_states, dones

      def add(self, state, action, reward, next_state, done):
            exp = self.experience(state, action, reward, next_state, done)
            self.buffer.append(exp)
            

      def __len__(self):
            return len(self.buffer)