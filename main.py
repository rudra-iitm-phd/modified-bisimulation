import gymnasium as gym
from state_embedding_cnn import StateEmbedding
from critic import ValueNetwork
from actor import SacActor
from sac_agent import SacAgent
import torch
from distracting_control import suite
import numpy as np
from PIL import Image
import torchvision.transforms as T


EMBEDDING_DIM = 128
HIDDEN_DIM = 256
DEVICE = "cuda" if torch.cuda.is_available() else("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 128
GAMMA = 0.99
LEARNING_RATE = 3e-4
EPISODES = 2000
FRAME_STACK = 3
IMAGE_SIZE = (84, 84)  # Target size for resizing

class ImageProcessor:
      def __init__(self, target_size=(84, 84)):
            self.target_size = target_size
            self.transform = T.Compose([
                  T.ToPILImage(),
                  T.Resize(target_size),
                  T.Grayscale(num_output_channels=3),  # Keep 3 channels but convert if needed
                  T.ToTensor(),
            ])
      
      def process(self, image):
            """Convert (240, 320, 3) image to (3, 84, 84) tensor"""
            # image is (240, 320, 3) from DMControl
            if isinstance(image, np.ndarray):
                  # Normalize to [0, 1] if needed
                  if image.dtype == np.uint8:
                        image = image.astype(np.float32) / 255.0
                  tensor = self.transform(image)
                  return tensor
            return image

if __name__ == "__main__":

      env = suite.load(
            domain_name="cheetah",
            task_name="run",
            difficulty="hard",
            dynamic=True)

      ACTION_DIM = env.action_spec().shape[0]
      
      # Initialize image processor
      image_processor = ImageProcessor(IMAGE_SIZE)
      
      # Get state shape after processing
      dummy_state = env.reset()
      if isinstance(dummy_state, dict):
            dummy_pixels = dummy_state["pixels"]
      else:
            dummy_pixels = dummy_state
      
      processed_dummy = image_processor.process(dummy_pixels)
      STATE_SHAPE = (FRAME_STACK * 3, IMAGE_SIZE[0], IMAGE_SIZE[1])  # [channels * frame_stack, 84, 84]

      embedding_net = StateEmbedding(STATE_SHAPE, EMBEDDING_DIM, frame_stack=FRAME_STACK)
      embedding_net.to(DEVICE)

      critic = ValueNetwork(EMBEDDING_DIM, ACTION_DIM)
      critic.to(DEVICE)

      target_critic = ValueNetwork(EMBEDDING_DIM, ACTION_DIM)
      target_critic.to(DEVICE)

      actor = SacActor(EMBEDDING_DIM, ACTION_DIM, HIDDEN_DIM)
      actor.to(DEVICE)

      agent = SacAgent(
            embedding_net, critic, target_critic, actor, env, DEVICE, 
            BATCH_SIZE, GAMMA, 0.2, LEARNING_RATE, 
            frame_stack=FRAME_STACK,
            image_processor=image_processor
      )

      agent.train(EPISODES, False)