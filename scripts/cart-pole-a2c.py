"""
Cart-pole reinforcement learning environment:
Agent learns to balance a pole on a cart

a2c: Agent uses Advantage Actor Critic algorithm

"""
import gym
import torch
from src.a2c import A2C
import torch.optim as optim
from tqdm import tqdm


LR = .001  # Learning rate
SEED = 2  # Random seed for reproducibility

# Experiments:

# 0 --> Diverges
# 1 --> Solves after 1600 episodes
# 2 --> Solves after 900 episdoes
# 3 --> Diverges

torch.manual_seed(SEED)
agent = A2C(gym.make('CartPole-v0'), random_seed=SEED)

#
# Train
#

r = []
avg_r = 0
i = 0

actor_optim = optim.Adam(agent.get_actor_params(), lr=LR)
critic_optim = optim.Adam(agent.get_critic_params(), lr=LR)

for i in range(10000):
    critic_optim.zero_grad()
    actor_optim.zero_grad()

    rewards, critic_vals, action_p_vals, total_reward = agent.train_env_episode(render=False)
    r.append(total_reward)

    l_actor, l_critic = agent.compute_loss(action_p_vals=action_p_vals, G=rewards, V=critic_vals)

    l_actor.backward(retain_graph=True)
    l_critic.backward()

    actor_optim.step()
    critic_optim.step()

    # Check average reward every 100 episodes, print, and end script if solved
    episode_count = i-(i%100)
    if len(r) % 100 == 0 and len(r) != 0:  # check average every 100 episodes
        avg_r = sum(r[len(r)-100:])/len(r[len(r)-100:])
        print(f'Average reward during episodes {episode_count}-{episode_count+100} is {avg_r.item()}')
        if avg_r > 195:
            print(f"Solved CartPole-v0 with average reward {avg_r.item()}")
            break


#
# Test
#
for _ in range(100):
    agent.test_env_episode(render=True)
