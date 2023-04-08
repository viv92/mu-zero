import numpy as np
import gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random

# q function q(s)
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
env = gym.envs.make("MountainCar-v0")
obs_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

# hyperparams
num_episodes = 10000
lr = 1e-3
df = 0.99
eps = 1 # decayed
eps_decay_step = 0.01
num_prev_frames = 2

# init q function
qnet = Qnet(obs_dim * num_prev_frames, num_actions)

# optimizer
optimizer = torch.optim.Adam(qnet.parameters(), lr=lr)

# run
ep_undiscounted_returns = []
for ep in range(1, num_episodes):
    # decay eps
    eps_decay_epoch = num_episodes * eps_decay_step
    if ep % eps_decay_epoch == 0:
        eps -= eps_decay_step

    # reset env for new episode
    ep_undiscounted_return = 0
    # start state
    obs = env.reset()
    state = np.zeros(obs_dim * num_prev_frames)
    state = np.concatenate((obs, state[obs_dim:]), axis=0)

    while(True):
        # if ep % 100 == 0:
        #     env.render()

        # take epsilon greedy action
        q_values = qnet.forward(state)
        action = get_epsgreedy_action(q_values, num_actions, eps)
        next_obs, reward, done, _ = env.step(action)

        next_state = np.concatenate((next_obs, state[obs_dim:]))
        ep_undiscounted_return += reward

        # calculate td error
        next_q_values = qnet.forward(next_state)
        td_error = reward + df * torch.max(next_q_values) - q_values[action]

        # loss
        loss = td_error ** 2

        # train
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # for next step in episode
        state = next_state
        if done:
            if ep % 1000 == 0:
                print('ep:{} \t  undiscounted_return:{} \t loss:{:.3f}'.format(ep, ep_undiscounted_return, loss))
            ep_undiscounted_returns.append(ep_undiscounted_return)
            break


# plot
fig = plt.figure()
plt.plot(ep_undiscounted_returns)
plt.xlabel('episode')
plt.ylabel('undiscounted return')
plt.savefig('mcar_qlearning_prevFrames='+str(num_prev_frames)+'_epsDecay='+str(eps_decay_step)+'_lr='+str(lr)+'_numEP='+str(num_episodes)+'.png')
