import torch.nn as nn


class A2C(nn.Module):

    def __init__(self, env, n_step=1):
        """
        Assumes fixed continuous observation space
        and fixed discrete action space (for now)

        :param env: target gym environment
        :param n_step: n-steps for a2c
        """
        self.env = env
        self.n_step = n_step
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
