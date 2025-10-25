from critic import ValueNetwork
from actor import SacActor
from buffer import Buffer
from state_embedding import StateEmbedding

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque 
import numpy as np
import gymnasium as gym


class SacAgent:
      def __init__(self, embedding, critic, target_critic, actor, env, device, batch_size, gamma, alpha=0.2, lr=3e-4, target_entropy=None, frame_stack=3, image_processor=None):

            self.env = env
            self.critic = critic
            self.target_critic = target_critic
            self.actor = actor
            self.embedding = embedding
            self.device = device
            self.frame_stack = frame_stack
            self.image_processor = image_processor
            
            # Initialize buffer with frame stacking capability
            self.buffer = Buffer(1000000, frame_stack=frame_stack)
            
            self.batch_size = batch_size
            self.gamma = gamma
            self.alpha = alpha

            # Get action dimension from DMControl
            action_shape = env.action_spec().shape[0]
            if target_entropy is None:
                  self.target_entropy = -torch.prod(torch.Tensor([action_shape]).to(device)).item()
            else:
                  self.target_entropy = target_entropy

            self.actor_params = list(self.actor.parameters()) + list(self.embedding.parameters())
            self.critic_params = list(self.critic.parameters()) + list(self.embedding.parameters())
            self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=device, dtype=torch.float32)
            
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
            self.optimizer_actor = torch.optim.Adam(self.actor_params, lr=lr)
            self.optimizer_critic = torch.optim.Adam(self.critic_params, lr=lr)
            
            # Initialize frame stack for current episode
            self.current_frames = None

      def extract_pixels(self, observation):
            """Extract pixels from DMControl observation"""
            if isinstance(observation, dict):
                  return observation["pixels"]
            elif hasattr(observation, 'pixels'):
                  return observation.pixels
            else:
                  return observation

      def reset_frame_stack(self, first_observation):
            """Initialize frame stack with zeros and add first state"""
            # Extract pixels from DMControl observation
            first_pixels = self.extract_pixels(first_observation)

            # Process the image
            first_state = self.image_processor.process(first_pixels)  # Returns (3, 84, 84) tensor
            first_state_np = first_state # Convert to numpy for stacking
            
            self.current_frames = deque(maxlen=self.frame_stack)
            
            # Fill with zeros (except the last frame which will be the current state)
            for _ in range(self.frame_stack - 1):
                  zero_frame = np.zeros_like(first_state_np)
                  self.current_frames.append(zero_frame)
            
            # Add the current state as the most recent frame
            self.current_frames.append(first_state_np)
            
            # Stack frames along channel dimension
            return self._get_stacked_state()

      def update_frame_stack(self, new_observation):
            """Add new frame to stack and return stacked state"""
            new_pixels = self.extract_pixels(new_observation)
            new_state = self.image_processor.process(new_pixels)
            new_state_np = new_state.numpy()
            self.current_frames.append(new_state_np)
            return self._get_stacked_state()

      def _get_stacked_state(self):
            """Convert deque of frames to stacked tensor"""
            # Stack frames along channel dimension: [frame_stack, 3, H, W] -> [3 * frame_stack, H, W]
            stacked = np.concatenate(list(self.current_frames), axis=0)
            return stacked

      def compute_critic_loss(self, states, actions, next_states, rewards, dones):
            with torch.no_grad():
                  # Get next state embeddings and sample actions for next states
                  next_state_emb = self.embedding(next_states)
                  next_actions, next_log_probs, _ = self.actor.sample(next_state_emb)
                  
                  # Target Q-values
                  _, target_q1, target_q2 = self.target_critic(next_state_emb, next_actions)
                  target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
                  target_values = rewards + self.gamma * (1 - dones) * target_q.squeeze()
            
            # Current Q-values
            state_emb = self.embedding(states)
            current_q1, current_q2 = self.critic.get_q_values(state_emb, actions)
            
            # Critic loss
            q1_loss = F.mse_loss(current_q1.squeeze(), target_values)
            q2_loss = F.mse_loss(current_q2.squeeze(), target_values)
            critic_loss = q1_loss + q2_loss
            
            return critic_loss

      def compute_actor_loss(self, states):
            state_embeddings = self.embedding(states)
            actions, log_probs, _ = self.actor.sample(state_embeddings)

            # SAC style actor loss with Q-values
            with torch.no_grad():
                  current_q1, current_q2 = self.target_critic.get_q_values(state_embeddings, actions)
                  current_q = torch.min(current_q1, current_q2)
            
            actor_loss = (self.alpha * log_probs.squeeze() - current_q.squeeze()).mean()
            return actor_loss, log_probs

      def compute_alpha_loss(self, log_probs):
            """Compute loss for automatic entropy tuning"""
            alpha_loss = - (self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            return alpha_loss

      def soft_update(self, tau=0.005):
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                  target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

      def compute_bisimulation_loss(self, states_1, states_2, rewards_1, rewards_2):
            pass

      def train(self, n_episodes, w_bisim:bool):
            reward_history = deque(maxlen=100)

            for ep in range(n_episodes):
                  # DMControl reset returns a TimeStep object, not a simple state
                  timestep = self.env.reset()
                  
                  # Initialize frame stack for new episode
                  stacked_state = self.reset_frame_stack(timestep.observation)
                  
                  ep_reward = 0
                  done = False
                  
                  while not done:
                        state_tensor = torch.FloatTensor(stacked_state).unsqueeze(0).to(self.device)
                        state_embedding = self.embedding(state_tensor)

                        action, log_probs, _ = self.actor.sample(state_embedding)
                        action_np = action.cpu().detach().numpy()[0]

                        # DMControl step returns TimeStep object
                        timestep = self.env.step(action_np)
                        
                        # Extract reward and check if episode ended
                        reward = timestep.reward
                        done = timestep.last()
                        
                        # Update frame stack with new observation
                        stacked_next_state = self.update_frame_stack(timestep.observation)
                        
                        # Store in buffer
                        self.buffer.add(
                              torch.FloatTensor(stacked_state).clone(), 
                              action.squeeze(0).clone(),
                              reward, 
                              torch.FloatTensor(stacked_next_state).clone(), 
                              done
                        )

                        stacked_state = stacked_next_state
                        ep_reward += reward

                        if len(self.buffer) > self.batch_size:
                              states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
                              
                              states = states.to(self.device)
                              actions = actions.to(self.device)
                              rewards = rewards.to(self.device)
                              next_states = next_states.to(self.device)
                              dones = dones.to(self.device)

                              # Critic update
                              critic_loss = self.compute_critic_loss(states, actions, next_states, rewards, dones)
                              if w_bisim:
                                    bisim_loss = self.compute_bisimulation_loss(states, next_states, reward, reward)
                                    critic_loss = critic_loss + 0.4 * bisim_loss

                              self.optimizer_critic.zero_grad()
                              critic_loss.backward()
                              self.optimizer_critic.step()

                              print("yes")

                              # Actor update
                              actor_loss, prob_logits = self.compute_actor_loss(states)
                              self.optimizer_actor.zero_grad()
                              actor_loss.backward()
                              self.optimizer_actor.step()

                              print("yes")

                              # Alpha update
                              alpha_loss = self.compute_alpha_loss(prob_logits)
                              self.alpha_optimizer.zero_grad()
                              alpha_loss.backward()
                              self.alpha_optimizer.step()
                              self.alpha = self.log_alpha.exp().item()

                              self.soft_update()
                        
                  reward_history.append(ep_reward)
                  avg_reward = np.mean(reward_history)

                  if (ep+1) % 20 == 0:
                        print(f"Episode {ep+1}, Reward: {ep_reward:.2f}, Avg Reward (last 100 eps): {avg_reward:.2f}")

                              








                        


                  




                  

            

      
            

      