import gymnasium as gym
from state_embedding import StateEmbedding
from critic import ValueNetwork
from actor import SacActor
from sac_agent import SacAgent
import torch


EMBEDDING_DIM = 128
HIDDEN_DIM = 256
DEVICE = "cuda" if torch.cuda.is_available() else("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 128
GAMMA = 0.99
LEARNING_RATE = 3e-4
EPISODES = 2000

if __name__ == "__main__":

      env = gym.make('HalfCheetah-v5')
      ACTION_DIM = env.action_space._shape[0]
      STATE_DIM = env.observation_space.shape[0]

      embedding_net = StateEmbedding(STATE_DIM, EMBEDDING_DIM)
      embedding_net.to(DEVICE)

      critic = ValueNetwork(EMBEDDING_DIM)
      critic.to(DEVICE)

      target_critic = ValueNetwork(EMBEDDING_DIM)
      target_critic.to(DEVICE)

      actor = SacActor(EMBEDDING_DIM, ACTION_DIM, HIDDEN_DIM)
      actor.to(DEVICE)

      agent = SacAgent(embedding_net, critic, target_critic, actor, env, DEVICE, BATCH_SIZE, GAMMA, 0.2, LEARNING_RATE)

      agent.train(EPISODES, False)


