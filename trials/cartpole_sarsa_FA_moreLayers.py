import numpy as np
import gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random

# q function
class Qnet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.h_dim = in_dim
        self.fc1 = nn.Linear(in_dim, self.h_dim)
        self.fc3 = nn.Linear(self.h_dim, self.h_dim)
        self.fc2 = nn.Linear(self.h_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, s):
        s = torch.from_numpy(s).float()
        h1 = self.relu(self.fc1(s))
        h2 = self.relu(self.fc3(h1))
        q = self.relu(self.fc2(h2))
        return q


# function to get epsilon greedy action
def get_epsgreedy_action(q_values, num_actions, eps):
    action = torch.argmax(q_values)
    action = action.data.numpy()
    random_prob = random.uniform(0,1)
    if random_prob < eps:
        action = np.random.choice(num_actions)
    return action


# load environment
env = gym.envs.make("CartPole-v0")
obs_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

qnet = Qnet(obs_dim + 1, 1) # input = concate(state,action); output = q(s,a)

# hyperparams
num_episodes = 10000
lr = 1e-3
df = 0.99
eps = 1 # decayed
eps_decay_step = 0.01

# optimizer
optimizer = torch.optim.Adam(qnet.parameters(), lr=lr)

# run
ep_undiscounted_returns = []
for ep in range(num_episodes):
    # decay eps
    eps_decay_epoch = num_episodes * eps_decay_step
    if ep % eps_decay_epoch == 0:
        eps -= eps_decay_step

    # reset env for new episode
    state = env.reset()
    ep_undiscounted_return = 0

    # get epsilon greedy action for first state in episode
    q_values = torch.zeros(num_actions)
    for tmp_action in np.arange(num_actions):
        q_values[tmp_action] = qnet.forward(np.concatenate((state, [tmp_action])))
    action = get_epsgreedy_action(q_values, num_actions, eps)

    ep_mean_loss = 0
    ep_steps = 0

    while(True):
        # if ep % 50 == 0:
        #     env.render()

        qv = qnet.forward(np.concatenate((state, [action])))

        # take action
        next_state, reward, done, _ = env.step(action)
        ep_undiscounted_return += reward

        # get next epsilon greedy action
        next_q_values = torch.zeros(num_actions)
        for tmp_next_action in np.arange(num_actions):
            next_q_values[tmp_next_action] = qnet.forward(np.concatenate((next_state, [tmp_next_action])))
        next_action = get_epsgreedy_action(next_q_values, num_actions, eps)

        # calculate td error
        td_target = reward + df * next_q_values[next_action]
        td_error = td_target - qv

        # loss
        loss = td_error ** 2

        # episode mean loss
        # ep_steps += 1
        # ep_mean_loss += (loss - ep_mean_loss) / ep_steps

        # train
        optimizer.zero_grad()
        loss.backward(retain_graph=False)
        optimizer.step()

        # for next step in episode
        state = next_state
        action = next_action
        # q_values = next_q_values

        if done:
            if ep % 1000 == 0:
                print('ep:{} \t  undiscounted_return:{} \t loss:{:.3f}'.format(ep, ep_undiscounted_return, loss.data.numpy()[0]))
            ep_undiscounted_returns.append(ep_undiscounted_return)
            break


# plot
fig = plt.figure()
plt.plot(ep_undiscounted_returns)
plt.xlabel('episode')
plt.ylabel('undiscounted return')
plt.savefig('cartpole_sarsa_moreLayers_epsDecay='+str(eps_decay_step)+'_lr='+str(lr)+'_numEP='+str(num_episodes)+'.png')
