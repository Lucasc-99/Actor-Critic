"""
Cart-pole reinforcement learning environment:
Agent learns to balance a pole on a cart

a2c: Agent uses Advantage Actor Critic algorithm

"""
import gym
from src.a2c import A2C
import torch.optim as optim
from tqdm import tqdm



agent = A2C(gym.make('CartPole-v0'))

actor_optim = optim.Adam(agent.get_actor_params(), lr=.001)
critic_optim = optim.Adam(agent.get_critic_params(), lr=.001)

r = []
avg_r = 0
solved_cond = False
i = 0

while not solved_cond and i < 10000:
    critic_optim.zero_grad()
    actor_optim.zero_grad()

    rewards, critic_vals, action_p_vals, total_reward = agent.run_env_episode()
    r.append(total_reward)

    l_actor, l_critic = agent.compute_loss(action_p_vals=action_p_vals, G=rewards, V=critic_vals)

    l_actor.backward(retain_graph=True)
    l_critic.backward()

    actor_optim.step()
    critic_optim.step()

    if len(r) % 100 == 0 and len(r) != 0:  # check average every 100 episodes
        avg_r = sum(r[len(r)-100:])/len(r[len(r)-100:])
        print(avg_r)
    i += 1
print(f"Done with average reward {avg_r} in final 100 episodes")