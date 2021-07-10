"""
Cart-pole reinforcement learning environment:
Agent learns to balance a pole on a cart

a2c: Agent uses Advantage Actor Critic algorithm

"""
import gym
from src.a2c import A2C

agent = A2C(gym.make('CartPole-v0'))

agent.train_episode()

# loss and backprop