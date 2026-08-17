# News Recommendation System using Reinforcement Learning

This project is a personalized News Recommendation System that recommends news based on user interests and learns from user feedback.

The system combines **Content-Based Filtering** with **Reinforcement Learning**.

## Features

- Reads news from `news.csv`
- Creates users manually with different interests
- Uses TF-IDF to convert news titles into numerical vectors
- Uses Cosine Similarity to find relevant news
- Uses user feedback to improve recommendations
- Like = +5 reward
- Dislike = -5 reward
- Implements three Reinforcement Learning algorithms:
  - Q-Learning
  - SARSA
  - Deep Q-Network (DQN)

## Project Structure

    News-Recommendation-System/
    │
    ├── news.csv
    ├── common.py
    ├── q_learning.py
    ├── sarsa.py
    ├── dqn.py
    ├── main.py
    └── README.md

## User Interests

Users are manually created with one or more interests.

Example:

    user1 -> Artificial Intelligence
    user2 -> Stock Market, Crypto
    user3 -> Health
    user4 -> Politics, International
    user5 -> Energy
    user6 -> Robotics, Technology
    user7 -> Business
    user8 -> Cybersecurity, Artificial Intelligence

User interests are not limited to the original categories available in the dataset.

## Content-Based Filtering

TF-IDF converts news titles and user interest topics into numerical vectors.

Cosine Similarity is then used to find news articles that are most relevant to a selected topic.

Flow:

    User Interest
         ↓
       TF-IDF
         ↓
    Cosine Similarity
         ↓
    Relevant News

## Reinforcement Learning

The recommender learns from user feedback.

    Select User
         ↓
    Select Topic
         ↓
    Recommend News
         ↓
    Like / Dislike
       +5 / -5
         ↓
    Update Model
         ↓
    Better Recommendations

## Q-Learning

Q-Learning uses a Q-Table where:

- Rows represent users
- Columns represent topics
- Q-values represent learned preferences

Initially, all Q-values are 0.

The values are updated according to user rewards.

Q-Learning uses the **best possible next action** while updating its Q-values.

## SARSA

SARSA also uses a Q-Table.

The main difference is:

    Q-Learning -> Uses the best next action
    SARSA      -> Uses the actual next action selected

Q-Learning is off-policy while SARSA is on-policy.

## Deep Q-Network (DQN)

DQN replaces the Q-Table with a Neural Network.

    User State
         ↓
    Neural Network
         ↓
    Q-values for Topics
         ↓
    Best Topic

Instead of directly storing Q-values in a table, DQN learns neural network weights and predicts Q-values.

## Reward System

| Feedback | Reward |
|----------|--------|
| Like | +5 |
| Dislike | -5 |

## Exploration and Exploitation

The system uses an Epsilon-Greedy strategy.

- Exploitation: Select the topic with the highest learned Q-value
- Exploration: Try another topic to discover new user interests

## Algorithms Comparison

| Algorithm | Method |
|----------|--------|
| Q-Learning | Q-Table + Best Next Action |
| SARSA | Q-Table + Actual Next Action |
| DQN | Neural Network + Predicted Q-values |

## Installation

Install required libraries:

    pip install pandas numpy scikit-learn torch

## Run Project

Run:

    python main.py

Then select a model:

    1. Q-Learning
    2. SARSA
    3. DQN

Select a user and provide feedback for each recommended news article.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- PyTorch
- TF-IDF
- Cosine Similarity
- Reinforcement Learning

## Conclusion

This project demonstrates how Content-Based Filtering and Reinforcement Learning can work together to build an adaptive News Recommendation System.

TF-IDF and Cosine Similarity find relevant news articles, while Q-Learning, SARSA and DQN learn user preferences from feedback and improve future recommendations.