import math

import numpy as np
# import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import mean_squared_error
# from matplotlib import pyplot as plt


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.n_actions = action_size

    def create_model(self, lr=0.001, gamma=0.99, exploration_proba=1.0,
                 exploration_proba_decay=0.005, batch_size=32, max_memory_buffer=2000, lambda_regularization=0.5):
        # we define some parameters and hyperparameters:
        # "lr" : learning rate
        # "gamma": discounted factor
        # "exploration_proba_decay": decay of the exploration probability
        # "batch_size": size of experiences we sample to train the DNN
        self.gamma = gamma
        self.lr = lr
        self.exploration_proba = exploration_proba
        self.exploration_proba_decay = exploration_proba_decay
        self.batch_size = batch_size
        self.lambda_regularization = lambda_regularization
        # We define our memory buffer where we will store our experiences
        # We stores only the 2000 last time steps
        self.memory_buffer = list()
        self.max_memory_buffer = max_memory_buffer
        self.recent_victim_action = 0
        self.recent_target = 0

        # We creaate our model having to hidden layers of 24 units (neurones)
        # The first layer has the same size as a state size
        # The last layer has the size of actions space
        self.model = Sequential([
            Dense(units=24, input_dim=self.state_size, activation='relu'),
            Dense(units=24, activation='relu'),
            Dense(units=self.n_actions, activation='linear')
        ])
        # self.model.compile(loss=self.regularized_loss,    # "mse",
         #                  optimizer=Adam(lr=self.lr))

        self.optimizer = tf.keras.optimizers.Adam(lr=self.lr)
        self.test_mode = False

    def load_model(self):
        self.model = keras.models.load_model("Trained Models/OA/Advagent_regul")
        self.test_mode = True

    # The agent computes the action to perform given a state
    def compute_action(self, current_state, avail_actions):
        # We sample a variable uniformly over [0,1]
        # if the variable is less than the exploration probability
        #     we choose an action randomly
        # else
        #     we forward the state through the DNN and choose the action
        #     with the highest Q-value.
        avail_actions_ind = np.nonzero(avail_actions)[0]

        if not self.test_mode:
            if np.random.uniform(0, 1) < self.exploration_proba:
                return np.random.choice(avail_actions_ind)
        q_values = self.model.predict(current_state)[0]
        for ind in range(self.n_actions):
            if avail_actions[ind] == 0:
                q_values[ind] = -math.inf
        return np.argmax(q_values)

    # when an episode is finished, we update the exploration probability using
    # espilon greedy algorithm
    def update_exploration_probability(self):
        self.exploration_proba = self.exploration_proba * np.exp(-self.exploration_proba_decay)
        # print(self.exploration_proba)

    # At each time step, we store the corresponding experience
    def store_episode(self, current_state, action, reward, next_state, done, victim_action):
        # We use a dictionnary to store them
        self.memory_buffer.append({
            "current_state": current_state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "victim_action": victim_action
        })
        # If the size of memory buffer exceeds its maximum, we remove the oldest experience
        if len(self.memory_buffer) > self.max_memory_buffer:
            self.memory_buffer.pop(0)

    # At the end of each episode, we train our model
    def train(self):
        # We shuffle the memory buffer and select a batch size of experiences
        np.random.shuffle(self.memory_buffer)
        batch_sample = self.memory_buffer[0:self.batch_size]

        # We iterate over the selected experiences
        for experience in batch_sample:
            # We compute the Q-values of S_t
            q_current_state = self.model.predict(experience["current_state"])
            # We compute the Q-target using Bellman optimality equation
            q_target = experience["reward"]
            if not experience["done"]:
                q_target = q_target + self.gamma * np.max(self.model.predict(experience["next_state"])[0])
            q_current_state[0][experience["action"]] = q_target

            # train the model
            self.recent_target = experience["action"]
            self.recent_victim_action = np.array(experience["victim_action"])
            # print("Outfunc,{},{}".format(self.recent_target, self.recent_victim_action))

            self.fit(experience["current_state"], np.array(q_current_state))

    def regularized_loss(self, inputs, q_true):
        q_pred = self.model(inputs)
        term1 = tf.math.squared_difference(q_true, q_pred)[0][self.recent_target]
        term2 = tf.math.squared_difference(q_pred[0][self.recent_victim_action], q_pred[0][self.recent_target])
        # print("infunction,{},{}".format(term1, term2))
        loss = term1 + (self.lambda_regularization * term2)
        return loss

    def grad(self, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.regularized_loss(inputs, targets)
        return tape.gradient(loss_value, self.model.trainable_variables)

    def fit(self, inputs, q_true):
        grads = self.grad(inputs, q_true)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

