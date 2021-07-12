import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

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
            nn.Linear(8, self.out_size)
        ).double()

        self.critic = nn.Sequential(
            nn.Linear(self.in_size, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        ).double()

    def run_env_episode(self):
        """
        Runs one episode and collects critic values, expected return,
        :return: A tensor with total/expected reward, critic eval, and action information
        """
        rewards = []
        critic_vals = []
        action_p_vals = []

        # Run episode and save information

        observation = self.env.reset()

        for _ in range(self.t_max):
            # self.env.render()
            observation = torch.from_numpy(observation).double()

            # Get action from actor
            action_logits = torch.softmax(self.actor(observation), -1)
            action = Categorical(action_logits).sample()

            # Get action probability
            action_prob = action_logits[action]

            # Get value from critic
            pred = torch.squeeze(self.critic(observation).view(-1))

            # Write prediction and action/probabilities to arrays
            action_p_vals.append(action_prob)
            critic_vals.append(pred)

            # Send action to environment and get rewards, next state

            observation, reward, done, info = self.env.step(action.item())
            rewards.append(torch.tensor(reward).double())

            if done:
                break

        total_reward = sum(rewards)

        # Convert reward array to expected return
        for t_i in range(len(rewards)):
            for t in range(t_i + 1, len(rewards)):
                rewards[t_i] += rewards[t] * (self.gamma ** (t_i - t))

        # Convert output arrays to tensors using torch.stack
        def f(inp):
            return torch.stack(tuple(inp), 0)
        return f(rewards), f(critic_vals), f(action_p_vals), total_reward

    def zero_grad(self, set_to_none: bool = False):
        """
        Wrapper method for nn.Module.zero_grad(),
        Zeroes the gradients of the actor and critic networks
        :param set_to_none: set grads to none
        :return: None
        """
        self.actor.zero_grad(set_to_none)
        self.critic.zero_grad(set_to_none)

    def get_actor_params(self):
        """
        Wrapper method for actor params
        :return: Actor network parameters
        """
        return self.actor.parameters()

    def get_critic_params(self):
        """
        Wrapper method for critic params
        :return: Actor network parameters
        """
        return self.critic.parameters()

    @staticmethod
    def compute_loss(action_p_vals, G, V, loss=nn.SmoothL1Loss()):
        """
        Actor Advantage Loss, where advantage = G - V
        Critic Loss, using mean squared error
        :param loss: loss function for critic
        :param action_p_vals: Action Probabilities
        :param G: Expected Returns
        :param V: Predicted Values
        :return: Actor loss tensor, Critic loss tensor
        """
        assert len(action_p_vals) == len(G) == len(V)
        return -(torch.sum(torch.log(action_p_vals * (G - V)))), loss(G, V)
