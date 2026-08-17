import numpy as np
import random


class QLearning:

    def __init__(self, users, topics):

        self.Q = np.zeros(
            (len(users), len(topics))
        )

        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2


    def choose_action(self, user_index, user_interests, topics):

        # Exploration
        if random.random() < self.epsilon:
            return random.randrange(len(topics))

        # Initially all Q-values are zero
        if np.all(self.Q[user_index] == 0):

            interest = random.choice(user_interests)

            return topics.index(interest)

        # Exploitation
        return np.argmax(self.Q[user_index])


    def update(self, user_index, action, reward):

        old_q = self.Q[user_index][action]

        best_future_q = np.max(
            self.Q[user_index]
        )

        self.Q[user_index][action] = old_q + self.alpha * (
            reward
            + self.gamma * best_future_q
            - old_q
        )