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
      def __init__(self, embedding:StateEmbedding, critic:ValueNetwork, target_critic:ValueNetwork,  actor:SacActor, env:gym.Env, device:str, batch_size:int, gamma:float, alpha:float=0.2, lr:float=3e-4, target_entropy = None):

            self.env = env
            
            self.critic = critic
            self.target_critic = target_critic

            self.actor = actor

            self.embedding = embedding

            self.device = device

            self.buffer = Buffer(1000000)

            self.batch_size = batch_size

            self.gamma = gamma

            self.alpha = alpha

            if target_entropy is None:
                  self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
            else:
                  self.target_entropy = target_entropy

            self.actor_params = list(self.actor.parameters()) + list(self.embedding.parameters())
            self.critic_parmas = list(self.critic.parameters()) + list(self.embedding.parameters())
            self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=device, dtype=torch.float32)
            
            
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

            self.optimizer_actor = torch.optim.Adam( self.actor_params , lr=lr)
            self.optimizer_critic = torch.optim.Adam( self.critic_parmas, lr=lr)
            

      def compute_critic_loss(self, states, next_states, rewards, dones):
            with torch.no_grad():
                  target_values = self.target_critic(self.embedding(next_states)).squeeze()
                  y = rewards + self.gamma * (1 - dones) * target_values
            values = self.critic(self.embedding(states)).squeeze()
            return F.mse_loss(values, y)

      def compute_actor_loss(self, states):
            state_embeddings = self.embedding(states)
            actions, log_probs, _ = self.actor.sample(state_embeddings)

            # Value-based SAC style actor loss
            with torch.no_grad():
                  state_values = self.target_critic(state_embeddings).squeeze()

            actor_loss = (self.alpha * log_probs.squeeze() - state_values).mean()

            return actor_loss, log_probs

      def compute_alpha_loss(self, log_probs):
            """Compute loss for automatic entropy tuning"""
            alpha_loss = - (self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            return alpha_loss

      def compute_bisimulation_loss(self, states_1, states_2, rewards_1, rewards_2):
            embedding_1 = self.embedding(states_1)
            embedding_2 = self.embedding(states_2)

            embedding_diff = torch.norm(embedding_1 - embedding_2, dim = -1)

            with torch.no_grad():
                  value1 = self.target_critic(embedding_1)
                  value2 = self.target_critic(embedding_2)
                  value_diff = value1.squeeze() - value2.squeeze()

            reward_diff = rewards_1 - rewards_2

            value_norm = (torch.norm(value1, dim = -1) + torch.norm(value2, dim = -1)).mean(-1)

            kl_lower_bound = 0.5 * (torch.pow(value_diff - reward_diff, 2))/(self.gamma * (value_norm**2))

            return F.mse_loss(embedding_diff, kl_lower_bound)


      def soft_update(self, tau=0.005):
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                  target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

      def train(self, n_episodes, w_bisim:bool):

            reward_history = deque(maxlen=100)

            for ep in range(n_episodes):
                  state, _ = self.env.reset()
                  ep_reward = 0
                  done = False
                  
                  while not done:
                        state = torch.FloatTensor(state).to(self.device)
                        state_embedding = self.embedding(state)

                        action, log_probs, _ = self.actor.sample(state_embedding)

                        next_state, reward, terminated, truncated, _ = self.env.step(action.cpu().detach().numpy())

                        done = terminated or truncated

                        next_state = torch.FloatTensor(next_state)
                        
                        # store in buffer
                        self.buffer.add(state.detach(), action, reward, next_state, done)

                        state = next_state

                        ep_reward += reward

                        if len(self.buffer) > self.batch_size:

                              states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
                        
                              states = states.to(self.device)
                              actions = actions.to(self.device)
                              rewards = rewards.to(self.device)
                              next_states = next_states.to(self.device)
                              dones = dones.to(self.device)

                              critic_loss = self.compute_critic_loss(states, next_states, rewards, dones)

                              ## For Bisimulation loss

                              if w_bisim:
                                    states_2, _, rewards_2, _, _ = self.buffer.sample(self.batch_size)

                                    bisim_loss = self.compute_bisimulation_loss(states, states_2, rewards, rewards_2)

                                    critic_loss = critic_loss + 0.4 * bisim_loss

                              self.optimizer_critic.zero_grad()
                              critic_loss.backward()
                              self.optimizer_critic.step()


                              actor_loss, prob_logits = self.compute_actor_loss(states)

                              self.optimizer_actor.zero_grad()
                              actor_loss.backward()
                              self.optimizer_actor.step()

                              alpha_loss = self.compute_alpha_loss(prob_logits)
                              
                              self.alpha_optimizer.zero_grad()
                              alpha_loss.backward()
                              self.alpha_optimizer.step()
                              self.alpha = self.log_alpha.exp().item()

                              self.soft_update()
                  
                  # update reward history
                  reward_history.append(ep_reward)
                  avg_reward = np.mean(reward_history)

                  if (ep+1) % 20 == 0:
                         print(f"Episode {ep+1}, Avg Reward (last 100 eps): {avg_reward:.2f}")

                              








                        


                  




                  

            

      
            

      