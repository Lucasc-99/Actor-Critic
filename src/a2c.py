import torch
import torch.nn as nn
from IPython import display as ipythondisplay
from PIL import Image
from pyvirtualdisplay import Display
from torch.distributions import Categorical


class A2C(nn.Module):

    def __init__(self, env, hidden_size=128, gamma=.99, random_seed=None):
        """
        Assumes fixed continuous observation space
        and fixed discrete action space (for now)

        :param env: target gym environment
        :param gamma: the discount factor parameter for expected reward function :float
        :param random_seed: random seed for experiment reproducibility :float, int, str
        """
        super().__init__()

        if random_seed:
            env.seed(random_seed)
            torch.manual_seed(random_seed)

        self.env = env
        self.gamma = gamma
        self.hidden_size = hidden_size
        self.in_size = len(env.observation_space.sample().flatten())
        self.out_size = self.env.action_space.n

        self.actor = nn.Sequential(
            nn.Linear(self.in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.out_size)
        ).double()

        self.critic = nn.Sequential(
            nn.Linear(self.in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).double()

    def train_env_episode(self, render=False):
        """
        Runs one episode and collects critic values, expected return,
        :return: A tensor with total/expected reward, critic eval, and action information
        """
        rewards = []
        critic_vals = []
        action_lp_vals = []

        # Run episode and save information

        observation = self.env.reset()

        done = False
        while not done:

            if render:
                self.env.render()

            observation = torch.from_numpy(observation).double()

            # Get action from actor
            action_logits = self.actor(observation)

            action = Categorical(logits=action_logits).sample()

            # Get action probability
            action_log_prob = action_logits[action]

            # Get value from critic
            pred = torch.squeeze(self.critic(observation).view(-1))

            # Write prediction and action/probabilities to arrays
            action_lp_vals.append(action_log_prob)
            critic_vals.append(pred)

            # Send action to environment and get rewards, next state

            observation, reward, done, info = self.env.step(action.item())
            rewards.append(torch.tensor(reward).double())

        total_reward = sum(rewards)

        # Convert reward array to expected return and standardize
        for t_i in range(len(rewards)):

            for t in range(t_i + 1, len(rewards)):
                rewards[t_i] += rewards[t] * (self.gamma ** (t_i - t))

        # Convert output arrays to tensors using torch.stack
        def f(inp):
            return torch.stack(tuple(inp), 0)

        # Standardize rewards
        rewards = f(rewards)
        rewards = (rewards - torch.mean(rewards)) / (torch.std(rewards) + .000000000001)

        return rewards, f(critic_vals), f(action_lp_vals), total_reward

    def test_env_episode(self, render=True):
        """
        Run an episode of the environment in test mode
        :param render: Toggle rendering of environment :bool
        :return: Total reward :int
        """
        observation = self.env.reset()
        rewards = []

        if render:
            # GIF collection code
            screen = self.env.render(mode='rgb_array')
            im = Image.fromarray(screen)
            gif = [im]
            # GIF collection code

        done = False
        while not done:

            observation = torch.from_numpy(observation).double()

            # Get action from actor
            action_logits = self.actor(observation)
            action = Categorical(logits=action_logits).sample()

            observation, reward, done, info = self.env.step(action.item())
            rewards.append(reward)

            if render:
                # GIF collection code
                screen = self.env.render(mode='rgb_array')
                gif.append(Image.fromarray(screen))
                # GIF collection code



        return sum(rewards), gif

    @staticmethod
    def compute_loss(action_p_vals, G, V, critic_loss=nn.SmoothL1Loss()):
        """
        Actor Advantage Loss, where advantage = G - V
        Critic Loss, using mean squared error
        :param critic_loss: loss function for critic   :Pytorch loss module
        :param action_p_vals: Action Log Probabilities  :Tensor
        :param G: Actual Expected Returns   :Tensor
        :param V: Predicted Expected Returns    :Tensor
        :return: Actor loss tensor, Critic loss tensor  :Tensor
        """
        assert len(action_p_vals) == len(G) == len(V)
        advantage = G - V.detach()
        return -(torch.sum(action_p_vals * advantage)), critic_loss(G, V)
