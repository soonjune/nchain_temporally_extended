from nchain import NChainEnv
import numpy as np
import matplotlib.pyplot as plt
from learning_agents import QLearningAgent
from value_iteration_agent import NChainVIAgent
import numpy as np
from Q_learning_agent import Agent

if __name__ == '__main__':
    env = NChainEnv(max_steps=100)
    agent = Agent(lr=0.001, gamma=0.9, eps_start=1.0, eps_end=0.01,
    eps_dec=0.9999995, n_actions=2, n_states=10)

    n_games = 1000
    last_10_avgs = []
    rewards = []

    for i in range(n_games):
        done = False
        obs = env.reset()
        reward = 0
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            agent.learn(obs, action, reward, obs_)
            obs = obs_
        rewards.append(reward)

        if i % 100 == 0:
            average = np.mean(rewards[-10:])
            last_10_avgs.append(average)
    plt.plot(last_10_avgs)
    plt.show()

# # hyperparameters tuned for NChain
# epsilon = 0.1
# gamma = 0.9
# alpha = 0.1
#
# V = [0 for _ in range(env.n)]



# # this agent will default to going backwards (we know optimal policy is forward)
# ql_agent = QLearningAgent(env, training_iterations, testing_iterations, \
#                           epsilon, gamma, alpha)
#
# ql_agent.train_agent()
# ql_agent.test_nchain()
#
# print('Training with {} iterations...'.format(training_iterations))
#
# rewards = []
# env.reset()
# print(env.change, env.max_steps)
# for idx, a in enumerate([0]*(env.change) + [1]*(env.max_steps-env.change)):
#     action = a
#     print("action: ", action)
#     print("state: ", env.state)
#     if idx > 10 and reward != 10:
#         print("backwards")
#     state, reward, done, _ = env.step(action) # take a random action
#     rewards.append(reward)
#
# print(np.mean(rewards))
