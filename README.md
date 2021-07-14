
<p align="center">
  <h1 align="center">Actor-Critic</h1>
  <h3 align="center">The A2C Reinforcement Learning Method</h3>
</p>
<br />



<!-- ABOUT THE PROJECT -->
## Introduction
This project contains an implementation of the Advantage Actor-Critic Reinforcement Learning Method, and includes an example on Cart-Pole.
Cart-Pole is a game in which the player (in this case, our model) attempts to balance a pole on a cart. At each time step, the player can either accelerate the cart left or right uniformally. An episode of the game is lost if the pole falls + or - 15 degrees from vertical, and it is won if the player survives 200 time steps. 

In order to be considered a solution, an agent must survive an average of 195+ time steps over 100 episodes.

## Implementation Details



<!-- Results -->
## Results

<img src="https://github.com/Lucasc-99/Actor-Critic/blob/master/res/solved-cartpole-v0.gif" width="200" height="200" />

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

