import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


# =========================
# NEURAL NETWORK
# =========================

class Network(nn.Module):

    def __init__(self, state_size, action_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.ReLU(),

            nn.Linear(32, action_size)
        )


    def forward(self, x):
        return self.model(x)


# =========================
# DQN AGENT
# =========================

class DQN:

    def __init__(self, users, topics):

        self.users = users
        self.topics = topics

        self.state_size = len(users)
        self.action_size = len(topics)

        self.gamma = 0.9
        self.epsilon = 0.2
        self.lr = 0.001

        self.network = Network(
            self.state_size,
            self.action_size
        )

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.lr
        )

        self.loss_function = nn.MSELoss()


    # =========================
    # USER -> STATE VECTOR
    # =========================

    def get_state(self, user_index):

        state = np.zeros(self.state_size)

        state[user_index] = 1

        return torch.tensor(
            state,
            dtype=torch.float32
        )


    # =========================
    # CHOOSE ACTION
    # =========================

    def choose_action(
        self,
        user_index,
        user_interests,
        topics
    ):

        # Exploration
        if random.random() < self.epsilon:
            return random.randrange(
                self.action_size
            )

        state = self.get_state(user_index)

        # Q-values predicted by neural network
        with torch.no_grad():
            q_values = self.network(state)

        # Initially use user's manual interest
        if torch.all(q_values == 0):

            interest = random.choice(
                user_interests
            )

            return topics.index(interest)

        return torch.argmax(q_values).item()


    # =========================
    # LEARN
    # =========================

    def update(
        self,
        user_index,
        action,
        reward
    ):

        state = self.get_state(user_index)

        # Current predicted Q-values
        q_values = self.network(state)

        current_q = q_values[action]


        # Future best Q-value
        with torch.no_grad():

            next_q_values = self.network(state)

            best_future_q = torch.max(
                next_q_values
            )


        target = reward + (
            self.gamma * best_future_q
        )


        loss = self.loss_function(
            current_q,
            target
        )


        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()


    # =========================
    # GET Q VALUES
    # =========================

    def get_q_values(self, user_index):

        state = self.get_state(user_index)

        with torch.no_grad():

            q_values = self.network(state)

        return q_values.numpy()