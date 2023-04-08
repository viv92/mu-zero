import numpy as np
import gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random

# random seeds
# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)

# q function: input (s,a) pair and output q(s,a)
class Qnet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim*2)
        self.fc2 = nn.Linear(in_dim*2, in_dim)
        self.fc3 = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, s):
        s = torch.from_numpy(s).float()
        h = self.relu(self.fc1(s))
        h = self.relu(self.fc2(h))
        q = self.relu(self.fc3(h))
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
num_episodes = 10000
lr = 1e-2
df = 0.99
eps = 0.9 # decayed
eps_decay_step = 0.01
num_prev_frames = 2

# init q function
qnet = Qnet(obs_dim * num_prev_frames + 1, 1) # since input = concat(state, action)

# optimizer
optimizer = torch.optim.Adam(qnet.parameters(), lr=lr)

# run
ep_returns = []
all_step_loss = []
all_ep_loss = []
for ep in range(1, num_episodes):
    # decay eps
    eps_decay_epoch = num_episodes * eps_decay_step
    if ep % eps_decay_epoch == 0:
        if eps > 0:
            eps -= eps_decay_step

    # reset env for new episode
    ep_return = 0
    ep_length = 0
    ep_sar = []

    # start state
    obs = env.reset()
    state = np.zeros(obs_dim * num_prev_frames)
    state = np.concatenate((obs, state[obs_dim:]), axis=0)

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

        # update episode return and store transition
        ep_return += (df ** ep_length) * reward
        ep_sar.append([state, action, reward])
        ep_length += 1

        # for next step in episode
        obs = next_obs
        state = np.concatenate((obs, state[obs_dim:]), axis=0)

        if done:
            if ep % 1000 == 0:
                print('ep:{} \t  ep_return:{}'.format(ep, ep_return))
            ep_returns.append(ep_return)
            break

    # MC incremental training (update qnet towards MC targets)
    ep_loss = []
    for state, action, reward in ep_sar:
        q_pred = qnet.forward(np.concatenate((state, [action])))
        q_target = ep_return
        optimizer.zero_grad()
        loss = (q_pred - q_target) ** 2
        loss.backward()
        optimizer.step()
        ep_loss.append(loss.item())
        all_step_loss.append(loss.item())
        ep_return = (ep_return - reward) / df
    mean_ep_loss = sum(ep_loss)//len(ep_loss)
    all_ep_loss.append(mean_ep_loss)



# plot
fig, ax = plt.subplots(1,3)
ax[0].plot(ep_returns)
ax[1].plot(all_step_loss)
ax[2].plot(all_ep_loss)
plt.savefig('cartpole_MC_incremental_bignet_prevFrames='+str(num_prev_frames)+'_epsDecay='+str(eps_decay_step)+'_lr='+str(lr)+'_numEP='+str(num_episodes)+'.png')
