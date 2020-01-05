'''
---------------------------------------------------------------
        Stock Trading Bot using Reinforcemnt Learning
---------------------------------------------------------------
Script for training and testing stock trading bot using Reinforcement Learning (Deep Q-Learning)

Use:
python stock_bot.py -m train -p True
(or)
python stock_bot.py -m test -p True
'''

# import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import itertools
import argparse
import re
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# function to read the dataset
def get_data():
  df = pd.read_csv('stock_data.csv')
  return df.values

# a multi-stock environment class
class MultiStockEnvironment:
  # This class mimics OpenAI Gym API
  def __init__(self, data, initial_investment=20000):
    # data
    self.stock_price_history = data
    self.n_step, self.n_stock = self.stock_price_history.shape

    # attributes
    self.initial_investment = initial_investment
    self.cur_step = None
    self.stock_owned = None
    self.stock_price = None
    self.cash_in_hand = None

    self.action_space = np.arange(3**self.n_stock)

    # action permutations, store a list of permutated action elements
    self.action_list = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))

    # calculate size of state
    self.state_dim = self.n_stock * 2 + 1

    self.reset()

  def reset(self):  
    # reset pointer to 0 and return initial state
    self.cur_step = 0
    self.stock_owned = np.zeros(self.n_stock)
    self.stock_price = self.stock_price_history[self.cur_step]
    self.cash_in_hand = self.initial_investment
    return self._get_obs()

  def step(self, action):
    # perform the trade, move pointer
    # calculate reward, next state, portfolio value, done
    assert action in self.action_space

    # get current value before performing action
    prev_val = self._get_val()

    # update price (go to the next day)
    self.cur_step = self.cur_step + 1
    self.stock_price = self.stock_price_history[self.cur_step]

    #perform the trade
    self._trade(action)

    # get the new value after performing trade
    cur_val = self._get_val()

    # reward, i.e, increase in portfolio value
    reward = cur_val - prev_val

    # set done flaf if we have run out of data
    done = self.cur_step==self.n_step-1

    # store the current value of portfolio
    info = {'cur_val': cur_val}

    return self._get_obs(), reward, done, info

  def _get_obs(self):
    obs = np.empty(self.state_dim)
    obs[:self.n_stock] = self.stock_owned
    obs[self.n_stock:2*self.n_stock] = self.stock_price
    obs[-1] = self.cash_in_hand
    return obs

  def _get_val(self):
    return self.stock_owned.dot(self.stock_price) + self.cash_in_hand

  def _trade(self, action):
      # index the action we want to perform
      # define action vector
      action_vector = self.action_list[action]

      # determine which stocks to buy or sell
      sell_index = []                                                           # stores index of stocks we want to sell
      buy_index = []                                                            # stores index of stocks we want to buy
      for i,a in enumerate(action_vector):
        if a==0:
          sell_index.append(i)
        elif a==2:
          buy_index.append(i)

      # sell any stocks (all shares of that stock) that we want to sell
      if sell_index:
        for i in sell_index:
          self.cash_in_hand = self.cash_in_hand + self.stock_price[i] * self.stock_owned[i]
          self.stock_owned[i] = 0

      # buy any stocks that we want to buy
      if buy_index:
        # buy one share in each stock until we run out of money
        buy_me = True
        while buy_me:
          for i in buy_index:
            if self.cash_in_hand > self.stock_price[i]:
              # buy one share of that stock
              self.stock_owned[i] = self.stock_owned[i] + 1
              self.cash_in_hand = self.cash_in_hand - self.stock_price[i]
            else:
              buy_me = False  

# Replay buffer class (memory)
class ReplayBuffer:
  def __init__(self, obs_dim, act_dim, size):
    # Pre-allocating fixed size arrays of respective dimensions mnetionned above as class data members
    self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)                 # to store state
    self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)                 # to store next state
    self.acts_buf = np.zeros(size, dtype=np.uint8)                              # to store actions
    self.rews_buf = np.zeros(size, dtype=np.float32)                            # to store rewards
    self.done_buf = np.zeros(size, dtype=np.uint8)                              # to store boolean values indicating done flag
    self.ptr = 0                                                                # initial value of pointer is 0, to use pointer to traverse through the queue from beginning
    self.size = 0                                                               # initial size of buffer
    self.max_size = size                                                        # maximum size of the buffer

  def store(self, obs, act, rew, next_obs, done):
    # store the values in their respective buffers
    self.obs1_buf[self.ptr] = obs
    self.obs2_buf[self.ptr] = next_obs
    self.acts_buf[self.ptr] = act
    self.rews_buf[self.ptr] = rew
    self.done_buf[self.ptr] = done
    # increment pointer to move to next position in each buffer so we can store the next values
    self.ptr = (self.ptr+1) % self.max_size                                     # modulo operation for incrementing the pointer since the queue is going to be circular
    self.size = min(self.size+1, self.max_size)                                 # denotes memory/space of the buffer

  def sample_batch(self, batch_size=32):
    # return all the stored values randomly whenever called during training
    idxs = np.random.randint(0, self.size, size=batch_size)
    return dict(s1=self.obs1_buf[idxs], s2=self.obs2_buf[idxs], a=self.acts_buf[idxs], r=self.rews_buf[idxs], d=self.done_buf[idxs])  

# Neural Network model
def action_NN(input_dim, n_action, n_hidden_layers=1, hidden_dim=32):
  # input layer
  i = tf.keras.layers.Input(shape=(input_dim,))
  x = i 

  # hidden layers
  for layer in range(n_hidden_layers):
    x = tf.keras.layers.Dense(hidden_dim, activation='relu')(x)

  # output layer
  x = tf.keras.layers.Dense(n_action)(x)

  # create the fuctional API model
  model = tf.keras.models.Model(inputs=i, outputs=x)

  # compile the model 
  model.compile(loss='mse', optimizer=tf.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False))

  # print the NN architecture of the model
  # print((model.summary()))

  return model

# Agent class
class DQNAgent(object):
  def __init__(self, state_size, action_size):
    self.state_size = state_size
    self.action_size = action_size
    self.memory = ReplayBuffer(state_size, action_size, size=500)
    self.gamma = 0.95                                                           # discount rate
    self.epsilon = 1.0                                                          # exploration rate
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.model = action_NN(state_size, action_size)

  def update_replay_buffer(self, state, action, reward, next_state, done):
    # store in replay buffer
    self.memory.store(state, action, reward, next_state, done)

  def act(self, state):
    # calculate Q(s, a), take argmax over a
    # random choice of action
    if np.random.rand() <= self.epsilon:
      return np.random.choice(self.action_size)

    # predict action  
    act_values = self.model.predict(state)
    # return predicted action
    return np.argmax(act_values[0])  

  def replay(self, batch_size=32):
    # sample from replay buffer, make input-target pairs
    # model.train_on_batch(inputs, targets)

    # check if the replay buffer contains sufficient data
    if self.memory.size < batch_size:
      return      

    # sample a batch of data from the replay memory
    minibatch = self.memory.sample_batch(batch_size)
    states = minibatch['s1']
    next_states = minibatch['s2']
    actions = minibatch['a']
    rewards = minibatch['r']
    done = minibatch['d']

    # calculate tentative target Q(s', a)
    target = rewards + self.gamma * np.max(self.model.predict(next_states), axis=1)

    # set target to be reward only since value of terminal state is always 0
    target[done] = rewards[done]

    # Update/Reinforce training on the NN to update the network for actions which were actually taken
    # set target to prediction of all values
    # change the targets for action taken Q(s, a)
    target_full = self.model.predict(states)
    target_full[np.arange(batch_size), actions] = target

    # run one training step
    self.model.train_on_batch(states, target_full)

    if self.epsilon > self.epsilon_min:
      self.epsilon = self.epsilon*self.epsilon_decay

  def load(self, name):
    self.model.load_weights(name)

  def save(self, name):
    self.model.save_weights(name)    

# to make necessary directories if they don't exist
def make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)

# to visualize rewards after training and testing
def plot_rewards(choice):
  a = np.load(f'{rewards_folder}/{choice}.npy')

  print(f"Average Reward: {a.mean():.2f} | Min: {a.min():.2f} | Max: {a.max():.2f}")

  plt.hist(a, bins=30)
  plt.title(choice)
  plt.show()

# function to standardize values
def get_scaler(env):
  # function to standardize values
  # returns scikit-learn's scaler object to scale the states
  states = []
  for _ in range(env.n_step):
    action = np.random.choice(env.action_space)
    state, reward, done, info = env.step(action)
    states.append(state)
    if done:
      break

  scaler = StandardScaler()
  scaler.fit(states)
  return scaler    

# function to play the game once when called
def play_one_episode(agent, env, is_train):
  state = env.reset()
  state = scaler.transform([state])
  done = False

  while not done:
    action = agent.act(state)
    next_state, reward, done, info = env.step(action)
    next_state = scaler.transform([next_state])
    if is_train == 'train':
      agent.update_replay_buffer(state, action, reward, next_state, done)
      agent.replay(batch_size)
    state = next_state

  return info['cur_val']


# main function
models_folder = 'trader_models/'
rewards_folder = 'trader_rewards/'
num_episodes = 200
batch_size = 32
initial_investment = 50000

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, required=True, help='Either "train" or "test"')
parser.add_argument('-p','--plot', type=bool, required=False, default=False, help='Either "True" or "False"')
args = parser.parse_args()

make_dir(models_folder)
make_dir(rewards_folder)

data = get_data()
n_timesteps, n_stocks = data.shape

choice = args.mode
visualize = args.plot

n_train = n_timesteps // 2

train_data = data[:n_train]
test_data = data[n_train:]

env = MultiStockEnvironment(train_data, initial_investment)
state_size = env.state_dim
action_size = len(env.action_space)
agent = DQNAgent(state_size, action_size)
scaler = get_scaler(env)

# create empty list for storing final value of portfolio at the end of an episode
portfolio_value = []

if choice=='test':
    # then load previous scaler
    with open(f'{models_folder}/standard_scaler.dat', 'rb') as f:
        scaler = pickle.load(f)

    # reinstantiate environment with test data
    env = MultiStockEnvironment(test_data, initial_investment) 

    # reset epsilon, epsilon should not be 1 and no need to run multiple episodes if epsilon is 0
    agent.epsilon = 0.01

    # load trained weights of the NN model
    agent.load(f'{models_folder}/nn_agent.h5')

# play game for num_episode times
for e in range(num_episodes):
    t0 = datetime.now()
    val = play_one_episode(agent, env, choice)
    dt = datetime.now() - t0
    print(f"Episode {e+1}/{num_episodes}\nEpisode end value: {val:.2f}  |  Duration: {dt}")
    portfolio_value.append(val)                  

if choice=='train':
    # save the NN model
    agent.save(f'{models_folder}/nn_agent.h5')

    # save the scaler
    with open(f'{models_folder}/standard_scaler.dat', 'wb') as f:
        pickle.dump(scaler, f)

# save portfolio value of each episode
np.save(f'{rewards_folder}/{choice}.npy', portfolio_value)

# visulaize rewards if specified
if visualize==True:
    plot_rewards(choice)