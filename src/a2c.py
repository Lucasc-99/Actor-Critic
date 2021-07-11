import torch
import torch.nn as nn
from torch.distributions import Categorical

'''
Run the agent on the environment to collect training data per episode.
Compute expected return at each time step.
Compute the loss for the actor and critic models.
Compute gradients and update network parameters.
Repeat 1-4 until either success criterion or max episodes has been reached.
'''


class A2C(nn.Module):

    def __init__(self, env, t_max=1000, gamma=.99):
        """
        Assumes fixed continuous observation space
        and fixed discrete action space (for now)

        :param env: target gym environment
        :param t_max: max time step for single episode
        """
        super(A2C, self).__init__()

        self.env = env
        self.t_max = t_max
        self.gamma = gamma
        self.in_size = len(env.observation_space.sample().flatten())
        self.out_size = self.env.action_space.n

        self.actor = nn.Sequential(
            nn.Linear(self.in_size, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.Linear(8, self.out_size)
        ).double()

        self.critic = nn.Sequential(
            nn.Linear(self.in_size, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        ).double()

    def train_episode(self):
        """
        Runs one episode and collects critic values, expected return,
        :return: A tensor with total/expected reward, critic eval, and action information
        """
        rewards = []
        critic_vals = []
        actions = []

        # Run episode and save information

        observation = self.env.reset()
        for _ in range(self.t_max):
            # self.env.render()

            # Get action from actor
            action_logits = torch.softmax(self.actor(torch.tensor(observation).double()), -1)
            action = Categorical(action_logits).sample()

            # Get value from critic
            predicted_val = self.critic(torch.tensor(observation).double())

            critic_vals.append(predicted_val)
            actions.append(action)

            # Send action to environment and get rewards, next state
            observation, reward, done, info = self.env.step(action.item())

            rewards.append(reward)

            if done:
                break

        total_reward = sum(rewards)

        # G_t = summation (from t_i = t to T) of (gamma^(t_i-t)* r_t_i)

        # TODO: make this differentiable
        for t_i in range(len(rewards)):
            for t in range(0, t_i):
                rewards[t_i] += rewards[t]*(self.gamma**(t_i - t))

        return rewards, critic_vals, actions, total_reward
