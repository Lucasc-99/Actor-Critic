"""
Lunar-lander reinforcement learning environment:
Agent learns to land spacecraft

Baseline: Agent selects moves at random

"""

import gym

env = gym.make('LunarLander-v2')


t_steps = []
for i_episode in range(1000):

    observation = env.reset()  # Get initial observation

    for t in range(100):

        env.render()

        action = env.action_space.sample()  # Get a random action (left or right)

        observation, reward, done, info = env.step(action)  # Get next step of the game
        print(reward)
        if done:
            t_steps.append(t + 1)
            break
    break
for t in t_steps:
    print(t)
env.close()
