"""
Lunar-lander reinforcement learning environment:
Agent learns to land spacecraft

a2c: Agent uses Advantage Actor Critic algorithm

"""
import gym
from src.a2c import A2C
import torch.optim as optim
import math

LR = .001  # Learning rate
SEED = None  # Random seed for reproducibility
MAX_EPISODES = 10000  # Max number of episodes

# Init actor-critic agent
agent = A2C(gym.make('LunarLander-v2'), random_seed=SEED, gamma=.999)


# Init optimizers
actor_optim = optim.Adam(agent.actor.parameters(), lr=LR)
critic_optim = optim.Adam(agent.critic.parameters(), lr=LR)

#
# Train
#

r = []  # Array containing total rewards
avg_r = 0  # Value storing average reward over last 100 episodes
max_r = -math.inf

for i in range(MAX_EPISODES):
    critic_optim.zero_grad()
    actor_optim.zero_grad()

    rewards, critic_vals, action_lp_vals, total_reward = agent.train_env_episode(render=False)

    r.append(total_reward)

    # Check if we won the game
    if total_reward >= 200:
        print("solved")
        break

    # Check average reward every 100 episodes
    if len(r) >= 100:
        episode_count = i - (i % 100)
        prev_episodes = r[len(r) - 100:]
        avg_r = sum(prev_episodes) / len(prev_episodes)
        if len(r) % 100 == 0:
            print(f'Average reward during episodes {episode_count}-{episode_count + 100} is {avg_r.item()}')

    l_actor, l_critic = agent.compute_loss(action_p_vals=action_lp_vals, G=rewards, V=critic_vals)

    l_actor.backward()
    l_critic.backward()

    actor_optim.step()
    critic_optim.step()

#
# Test
#
for _ in range(10):
    agent.test_env_episode(render=True)
