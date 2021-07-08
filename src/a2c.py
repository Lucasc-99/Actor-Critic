import torch.nn as nn


class A2C(nn.Module):

    def __init__(self, env, n):
        """
        Assumes continuous observation space
        and discrete action space (for now)

        :param env: gym environment
        :param n: n-steps for a2c
        """
        self.env = env
        self.n = n
        self.flat_input_size = len(env.observation_space.sample().flatten())
        self.flat_output_size = env.action_space.n

        self.actor = nn.Sequential(
            nn.Linear(self.flat_input_size, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.Linear(8, env.action_space.n)
        )

        self.critic = nn.Sequential(
            nn.Linear(self.flat_input_size, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

