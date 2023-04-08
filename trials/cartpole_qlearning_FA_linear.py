import numpy as np
import gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random

# q function - input = concat(s,a); output = q(s,a)
class Qnet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.h_dim = in_dim
        self.fc1 = nn.Linear(in_dim, self.h_dim)
        self.fc2 = nn.Linear(self.h_dim, out_dim)

    def forward(self, s): # NOTE: no non-linearity since convergence results hold only for linear function approximator
        s = torch.from_numpy(s).float()
        h = self.fc1(s)
        q = self.fc2(h)
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

# init q function
qnet = Qnet(obs_dim+1, 1)  # since input = concat(s,a); output = q(s,a)

# hyperparams
num_episodes = 100000
lr = 1e-4
df = 0.99
eps = 1 # decayed
eps_decay_step = 0.01

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
    state = env.reset()
    ep_undiscounted_return = 0

    while(True):
        # if ep % 100 == 0:
        #     env.render()

        # get epsilon greedy action
        q_values = torch.zeros(num_actions)
        for tmp_action in np.arange(num_actions):
            q_values[tmp_action] = qnet.forward(np.concatenate((state, [tmp_action])))
        action = get_epsgreedy_action(q_values, num_actions, eps)

        # take action
        next_state, reward, done, _ = env.step(action)
        ep_undiscounted_return += reward

        # calculate td error - TODO: no_grad for next_q_values ?
        with torch.no_grad():
            next_q_values = torch.zeros(num_actions)
            for tmp_action in np.arange(num_actions):
                next_q_values[tmp_action] = qnet.forward(np.concatenate((next_state, [tmp_action])))
            td_target = reward + df * torch.max(next_q_values)

        td_error = td_target - q_values[action]

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
plt.savefig('cartpole_qlearning_linearFA_epsDecay='+str(eps_decay_step)+'_lr='+str(lr)+'.png')
