from common import (
    users,
    user_names,
    topics,
    get_article,
    get_reward
)

from q_learning import QLearning
from sarsa import SARSA
from dqn import DQN


# =========================
# SELECT MODEL
# =========================

print("\n1. Q-Learning")
print("2. SARSA")
print("3. DQN")

choice = input("\nSelect Model: ")


if choice == "1":

    model = QLearning(
        user_names,
        topics
    )

    model_name = "Q-Learning"


elif choice == "2":

    model = SARSA(
        user_names,
        topics
    )

    model_name = "SARSA"


else:

    model = DQN(
        user_names,
        topics
    )

    model_name = "DQN"


# =========================
# SELECT USER
# =========================

print("\nAvailable Users:")

for user in user_names:
    print(
        user,
        "->",
        users[user]
    )


user = input("\nSelect User: ")

while user not in users:
    user = input(
        "Invalid User. Select again: "
    )


user_index = user_names.index(user)

seen_articles = set()


# =========================
# RECOMMENDATION LOOP
# =========================

for episode in range(10):

    action = model.choose_action(
        user_index,
        users[user],
        topics
    )

    selected_topic = topics[action]


    article = get_article(
        selected_topic,
        seen_articles
    )


    if article is None:
        break


    print("\n================================")
    print("Recommendation:", episode + 1)
    print("Model:", model_name)
    print("RL Topic:", selected_topic)

    print(
        "Title:",
        article["Title"]
    )

    print(
        "Dataset Category:",
        article["Final_Category"]
    )

    print(
        "URL:",
        article["URL"]
    )


    # =========================
    # REWARD
    # =========================

    reward = get_reward()

    print("Reward:", reward)


    # =========================
    # UPDATE MODEL
    # =========================

    if model_name == "Q-Learning":

        model.update(
            user_index,
            action,
            reward
        )


    elif model_name == "SARSA":

        next_action = model.choose_action(
            user_index,
            users[user],
            topics
        )

        model.update(
            user_index,
            action,
            reward,
            next_action
        )


    else:

        model.update(
            user_index,
            action,
            reward
        )


# =========================
# FINAL RESULT
# =========================

print("\n============================")
print("LEARNED PREFERENCES")
print("============================")


if model_name == "DQN":

    values = model.get_q_values(
        user_index
    )

else:

    values = model.Q[user_index]


for topic, value in zip(
    topics,
    values
):

    print(
        topic,
        ":",
        round(float(value), 2)
    )