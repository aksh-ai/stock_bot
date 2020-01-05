'''
---------------------------------------------------------------
        Stock Trading Bot using Reinforcemnt Learning
---------------------------------------------------------------
Script to visualize the rewards for the stock trading bot

Use:
python vis_rewards.py -m train
(or)
python vis_rewards.py -m test
'''

# import necessary libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt

# function to plot rewards obtained during training or testing process
def plot_rewards(choice):
  a = np.load(f'{rewards_folder}/{choice}.npy')

  print(f"Average Reward: {a.mean():.2f} | Min: {a.min():.2f} | Max: {a.max():.2f}")

  plt.hist(a, bins=30)
  plt.title(choice)
  plt.show()

# main function
rewards_folder = 'trader_rewards/'

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, required=True, help='Either "train" or "test"')
args = parser.parse_args()

choice = args.mode

#plot the rewards
plot_rewards(choice)
