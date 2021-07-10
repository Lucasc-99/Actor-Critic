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
        self.env = env
        self.t_max = t_max
        self.gamma = gamma
        self.in_size = len(env.observation_space.sample().flatten())
        self.out_size = env.action_space.n

        self.actor = nn.Sequential(
            nn.Linear(self.in_size, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.Linear(8, self.out_size)
        )

        self.critic = nn.Sequential(
            nn.Linear(self.in_size, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def run_episode(self):
        """
        Runs one episode and collects critic values, expected return,
        :return: A pytorch tensor stacked with reward, critic eval, and action information
        """
        rewards = []
        critic_vals = []
        actions = []

        # Run episode and save information

        observation, reward = self.env.reset()
        for _ in range(self.t_max):
            # self.env.render()

            action_logits = torch.softmax(self.actor(observation))
            action = Categorical(action_logits).sample()

            predicted_val = self.critic(observation)

            rewards.append(reward)
            critic_vals.append(predicted_val)
            actions.append(action)

            observation, reward, done, info = self.env.step(action)

            if done:
                break

        # Convert observed rewards to expected
        running_sum = 0
        #for i in range(len(rewards)):
         #   rewards +=


        #return torch.stack()
