
<p align="center">
  <h1 align="center">Actor-Critic</h1>
  <h3 align="center">The A2C Reinforcement Learning Method</h3>
</p>
<br />



<!-- ABOUT THE PROJECT -->
## Introduction
This project contains an implementation of the Advantage Actor-Critic Reinforcement Learning Method, and includes an example on Cart-Pole.
Cart-Pole is a game in which the player (in this case, our agent) attempts to balance a pole on a cart. At each time step, the player can either accelerate the cart left or right uniformally. An episode of the game is lost if the pole falls + or - 15 degrees from vertical, and it is won if the player survives 200 time steps. 

In order to be considered a solution, the agent must survive an average of 195+ time steps over 100+ episodes.


## Implementation Details

At each time step, the agent provides an action to the environment and the environment provides an observation and a reward. In the case of Cart-Pole the reward at each time step is 1, such that the total reward for each episode depends on how long the agent survives the game. An observation is an array consisting of the following: (cart position, cart velocity, pole angle, pole rotation rate).

This implementation of the A2C method uses two neural networks:

 </br>
 Actor: takes in an observation as input and outputs action probabilities
 
 ```
 self.actor = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        ).double()
 ```
 </br>
 </br>
 Critic: takes in an observation and outputs a value which estimates the expected return at the current state
 
 ```
 self.critic = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).double()
 ```
 </br>
  
  The above code creates the network architectures for Cart-Pole, however the actual module in src/a2c.py infers the input and output dimensions and thus can be used for any OpenAI Gym Env

<!-- Results -->
## Results

Side-by-side comparison of random agent (takes random actions) and trained A2C agent:

<p float="left">  
    <img src="https://github.com/Lucasc-99/Actor-Critic/blob/master/res/unsolved-cartpole-v0_2.gif" width="300" height="200" />
    <img src="https://github.com/Lucasc-99/Actor-Critic/blob/master/res/solved-cartpole-v0_1.gif" width="300" height="200" />
</p>


Rewards at each episode for 4 seperate trials:

<p float="left">
   <img src="https://github.com/Lucasc-99/Actor-Critic/blob/master/res/cartpoledata_1.png" width="400" height="300" />
   <img src="https://github.com/Lucasc-99/Actor-Critic/blob/master/res/cartpoledata_2.png" width="400" height="300" />
   <img src="https://github.com/Lucasc-99/Actor-Critic/blob/master/res/cartpoledata_3.png" width="400" height="300" />
  <img src="https://github.com/Lucasc-99/Actor-Critic/blob/master/res/cartpoledata_5.png" width="400" height="300" />
</p>

Training can be somewhat unstable, but will usually converge well before 2000 episodes


## Built With

* [Pytorch](https://pytorch.org/)
* [Open AI Gym](https://gym.openai.com/)


<!-- Usage -->

## Installation and Running Scripts

1. Clone the repo and change into directory
   ```sh
   git clone https://github.com/Lucasc-99/Actor-Critic.git
   cd Actor-Critic
   ```
   
2. Install Pytorch and Gym
   ```sh
   pip3 install torch
   pip3 install gym
   ```
 
3. Run scripts
   ```sh
   python3 -m src.cart-pole-baseline.py
   python3 -m src.cart-pole-a2c.py
   ```

