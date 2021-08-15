from nchain import NChainEnv
from learning_agents import QLearningAgent
from value_iteration_agent import NChainVIAgent
import numpy as np

env = NChainEnv(max_steps=100)
env.reset()
training_iterations = 1
testing_iterations = 1000

# hyperparameters tuned for NChain
epsilon = 0.1
gamma = 0.9
alpha = 0.1

# this agent will default to going backwards (we know optimal policy is forward)
ql_agent = QLearningAgent(env, training_iterations, testing_iterations, \
                          epsilon, gamma, alpha)

ql_agent.train_agent()
ql_agent.test_nchain()

print('Training with {} iterations...'.format(training_iterations))

rewards = []
env.reset()
print(env.change, env.max_steps)
for idx, a in enumerate([0]*(env.change) + [1]*(env.max_steps-env.change)):
    action = a
    print("action: ", action)
    print("state: ", env.state)
    if idx > 10 and reward != 10:
        print("backwards")
    state, reward, done, _ = env.step(action) # take a random action
    rewards.append(reward)

print(np.mean(rewards))
