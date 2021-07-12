"""
Cart-pole reinforcement learning environment:
Agent learns to balance a pole on a cart

a2c: Agent uses Advantage Actor Critic algorithm

"""
import gym
import torch.optim as optim
from src.a2c import A2C

agent = A2C(gym.make('CartPole-v0'))

actor_optim = optim.SGD(agent.get_actor_params(), lr=.001)
critic_optim = optim.SGD(agent.get_critic_params(), lr=.001)

max_reward = 0

for episode in range(1000):
    actor_optim.zero_grad()
    critic_optim.zero_grad()

    rewards, critic_vals, action_p_vals, total_reward = agent.run_env_episode()

    max_reward = max(max_reward, total_reward)

    l_actor, l_critic = agent.compute_loss(action_p_vals=action_p_vals, G=rewards, V=critic_vals)

    l_actor.backward()
    l_critic.backward()

    actor_optim.step()
    critic_optim.step()
print(max_reward)