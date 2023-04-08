import numpy as np
import gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random

# q function: input (s,a) pair and output q(s,a)
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



# hyperparams
num_episodes = 100000
lr = 1e-4
df = 0.99
eps = 1 # decayed
eps_decay_step = 0.01
num_prev_frames = 4

# init q function
qnet = Qnet(obs_dim * num_prev_frames + 1, 1) # since input = concat(state, action)

# optimizer
optimizer = torch.optim.Adam(qnet.parameters(), lr=lr)

# run
ep_returns = []
all_loss = []
for ep in range(1, num_episodes):
    # decay eps
    eps_decay_epoch = num_episodes * eps_decay_step
    if ep % eps_decay_epoch == 0:
        eps -= eps_decay_step

    # reset env for new episode
    ep_return = 0
    ep_length = 0
    ep_sar = []
    # first state
    obs = env.reset()
    state = np.zeros(obs_dim * num_prev_frames)
    state = np.concatenate((obs, state[obs_dim:]))

    # MC - episode
    while(True):
        # if ep % 100 == 0:
        #     env.render()

        # get epsilon greedy action
        q_values = torch.zeros(num_actions)
        for tmp_action in np.arange(num_actions):
            q_values[tmp_action] = qnet.forward(np.concatenate((state, [tmp_action])))
        action = get_epsgreedy_action(q_values, num_actions, eps)

        # take action
        next_obs, reward, done, _ = env.step(action)
        next_state = np.concatenate((next_obs, state[obs_dim:]))

        # update episode return and store transition
        ep_return += (df ** ep_length) * reward
        ep_sar.append([state, action, reward])
        ep_length += 1

        # for next step in episode
        state = next_state
        if done:
            if ep % 1000 == 0:
                print('ep:{} \t  ep_return:{}'.format(ep, ep_return))
            ep_returns.append(ep_return)
            break

    # MC batched training (update qnet towards MC targets)
    ep_total_loss = 0
    for state, action, reward in ep_sar:
        q_pred = qnet.forward(np.concatenate((state, [action])))
        q_target = ep_return
        loss = (q_pred - q_target) ** 2
        ep_total_loss += loss
        ep_return = (ep_return - reward) / df

    optimizer.zero_grad()
    ep_total_loss.backward()
    optimizer.step()
    all_loss.append(ep_total_loss.item())





# plot
fig, ax = plt.subplots(1,2)
ax[0].plot(ep_returns)
ax[1].plot(all_loss)
plt.savefig('cartpole_MC_batched_prevFrames='+str(num_prev_frames)+'_epsDecay='+str(eps_decay_step)+'_lr='+str(lr)+'.png')
