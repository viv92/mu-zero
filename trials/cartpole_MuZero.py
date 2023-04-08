import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

# torch.autograd.set_detect_anomaly(True)

# H function - representation function to obtain abstract planning state from observation history
# todo - check if we should input action history along with observation history
# as of now - input = current observation, output = current abstract state s0 to start planning
class H_net(nn.Module):
    def __init__(self, in_dim, h_dim, nlayers=1):
        super().__init__()
        # lstm here
        self.lstm = nn.LSTM(in_dim, h_dim, nlayers)
        self.hstate = torch.zeros(h_dim) # initial hidden state of lstm
        self.cstate = torch.zeros(h_dim) # initial cell state of lstm
        self.h_dim = h_dim

    # reset hidden and cell states of the lstm to zeros - used at start of each episode
    def reset(self):
        self.hstate = torch.zeros(1, 1, self.h_dim)
        self.cstate = torch.zeros(1, 1, self.h_dim)

    # forward prop through lstm
    def forward(self, obs_seq):
        obs_seq = np.array(obs_seq)
        obs_seq = torch.from_numpy(obs_seq).unsqueeze(0)
        # s0, (hn, cn) = self.lstm(obs_seq, (self.hstate, self.cstate))
        s0, (hn, cn) = self.lstm(obs_seq)
        self.hstate, self.cstate = hn, cn
        s0 = s0.squeeze(0) # remove batch dimension
        s0 = s0[-1] # pick the last output of the lstm as the state embedding
        return s0



# G function - for modeling planning state transitions and rewards
# fully connected network with two output heads - one for state transitions and other for rewards
# input to net - concat(state, action)
class G_net(nn.Module):
    def __init__(self, s_dim, h_dim):
        super().__init__()
        self.fc0 = nn.Linear(s_dim+1, h_dim)
        self.fc1 = nn.Linear(h_dim, h_dim)
        self.state_fc0 = nn.Linear(h_dim, h_dim)
        self.state_fc1 = nn.Linear(h_dim, s_dim)
        self.reward_fc0 = nn.Linear(h_dim, h_dim)
        self.reward_fc1 = nn.Linear(h_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        sa = torch.cat((state, action.unsqueeze(0)))
        h1 = self.relu(self.fc0(sa))
        h2 = self.relu(self.fc1(h1))
        h_state = self.relu(self.state_fc0(h2))
        next_state = self.relu(self.state_fc1(h_state))
        h_reward = self.relu(self.reward_fc0(h2))
        reward = self.relu(self.reward_fc1(h_reward))
        return next_state, reward[0]


# F function - prediction function to predict state value and simulation policy
# fully connected network with two heads - one for state value and other for policy
# input to network = abstract state s_k
# output = (v_k, p_k) : state value and simulation policy for state s_k
class F_net(nn.Module):
    def __init__(self, s_dim, h_dim, a_dim): # a_dim = num_actions
        super().__init__()
        self.fc0 = nn.Linear(s_dim, h_dim)
        self.fc1 = nn.Linear(h_dim, h_dim)
        self.v_fc0 = nn.Linear(h_dim, h_dim)
        self.v_fc1 = nn.Linear(h_dim, 1)
        self.p_fc0 = nn.Linear(h_dim, h_dim)
        self.p_fc1 = nn.Linear(h_dim, a_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, state):
        h1 = self.relu(self.fc0(state))
        h2 = self.relu(self.fc1(h1))
        h_v = self.relu(self.v_fc0(h2))
        v = self.relu(self.v_fc1(h_v))
        h_p = self.relu(self.p_fc0(h2))
        p = self.relu(self.p_fc1(h_p))
        p = self.softmax(p)
        return v[0], p


# MCTS node - represents a node in MCTS tree
class MCTS_node():
    def __init__(self, state, parent, num_actions, v, p):
        self.state = state # abstract state
        self.prior_v = v # predicted state value
        self.prior_p = p # predicted simulation policy
        self.parent = parent # parent node
        self.expanded = False # no action expanded from this node (yet)
        self.children = {} # key = action, value = child_node - stores child nodes that are part of the search tree
        self.visit_count = 0 # used for computing ucb score for action selection; for backing up value estimates from search; and for calculating pi_stats for the root node
        self.value = 0 # estimated value of this node (obtained from mcts)


# MCTS
class MCTS():
    def __init__(self, obs_seq, hnet, fnet, gnet, num_actions, eps, df, dirichlet_alpha, plan_iters):
        self.hnet = hnet
        self.fnet = fnet
        self.gnet = gnet
        self.plan_iters = plan_iters
        self.num_actions = num_actions
        self.eps = eps
        self.df = df
        self.dirichlet_alpha = dirichlet_alpha
        # root node
        self.s0 = None
        # add root node to instantiate search tree
        self.add_root_node(obs_seq)

    # function to add root node - instantiates the search tree
    def add_root_node(self, obs_seq):
        # abstract state
        state = self.hnet.forward(obs_seq)
        # predicted state value and simulation policy
        v, p = self.fnet.forward(state)
        # instantiate
        self.s0 = MCTS_node(state, None, self.num_actions, v, p)

    # function to add a new (child) node to the search tree
    # returns the transition reward and the new state value to bootstrap the return estimates
    def add_node(self, parent, action):
        # get the (child) node's state and reward
        state, reward = self.gnet.forward(parent.state, action)
        # predict state value and simulation policy
        v, p = self.fnet.forward(state)
        # instantiate and add to search tree
        node = MCTS_node(state, parent, self.num_actions, v, p)
        parent.children[action.item()] = node
        return reward, v


    # function to backup search estimates up the tree (till the root)
    def backup_estimates(self, trajectory, bootstrap_value):
        state_return = bootstrap_value
        for node, action, reward in reversed(trajectory):
            node.visit_count += 1
            state_return = reward + self.df * state_return
            node.value += (1/node.visit_count) * (state_return - node.value)

    # function to select action when in the tree
    # action chosen based on ucb scores of valid actions from the node,
    # if root node, add dirichlet noise for exploration
    # can be thought of as representing the tree policy in mcts
    def get_noisyUCB_action(self, node):
        # get ucb score for each action
        ucb_scores = torch.zeros(self.num_actions)
        for action in torch.arange(self.num_actions):
            # proceed if child node corresponding to this action exists in the tree
            if action.item() in node.children:
                action_prior_prob = node.prior_p[action]
                ucb_score = action_prior_prob * np.math.sqrt(node.visit_count) / (node.children[action.item()].visit_count+1)
                if node.children[action.item()].visit_count > 0:
                    ucb_score += node.children[action.item()].value
                ucb_scores[action] = ucb_score
        # normallze ucb_scores
        scores_sum = torch.sum(ucb_scores)
        for a in torch.arange(self.num_actions):
            ucb_scores[a] /= scores_sum

        # if root node - add dirichlet noise
        if node == self.s0:
            d_alpha = torch.ones(self.num_actions) * self.dirichlet_alpha
            noise = np.random.dirichlet(alpha=d_alpha)
            noise = torch.from_numpy(noise)
            ucb_scores += noise
            # re-normalize ucb_scores
            scores_sum = torch.sum(ucb_scores)
            for a in torch.arange(self.num_actions):
                ucb_scores[a] /= scores_sum

        # return greedy action according to ucb_scores
        chosen_action = torch.argmax(ucb_scores)
        return chosen_action

    # function to select action when in the tree
    # action chosen based on visit counts of child nodes,
    # if root node, take eps greedy action for exploration
    # can be thought of as representing the tree policy in mcts
    def get_epsgreedy_visitcounts_action(self, node):
        # get visit score for each action
        visit_scores = torch.zeros(self.num_actions)
        for action in torch.arange(self.num_actions):
            # proceed if child node corresponding to this action exists in the tree
            if action.item() in node.children:
                visit_scores[action] = node.children[action.item()].visit_count

        # chosen action is the greedy action according to visit counts
        chosen_action = torch.argmax(visit_scores)

        # if root node - take eps greedy action
        if node == self.s0:
            random_prob = torch.distributions.uniform.Uniform(0,1).sample()
            if random_prob < self.eps:
                chosen_action = torch.randint(self.num_actions, (1,)).squeeze()

        return chosen_action


    # function to get pi_stats and value of the root node at the end of mcts - based on visit counts of chidren of the root node
    def get_pi_stats_and_value(self):
        children_visit_counts = torch.zeros(self.num_actions)
        for action in torch.arange(self.num_actions):
            if action.item() in self.s0.children:
                children_visit_counts[action] = self.s0.children[action.item()].visit_count
            else:
                children_visit_counts[action] = 0
        # normalize to get probabilities
        sum_counts = torch.sum(children_visit_counts)
        for a in torch.arange(self.num_actions):
            children_visit_counts[a] /= sum_counts
        pi_stats = children_visit_counts
        return pi_stats, self.s0.value

    # function to perform mcts
    # according to appendix B of the MuZero paper
    def mcts(self):
        for iter in range(self.plan_iters):

            trajectory = [] # container to store (state, action, reward) tuples constituting the planning trajectory
            sk = self.s0 # start at root node
            plan_iter_done = False
            while not plan_iter_done:
                if sk.expanded: # node has been expanded atleast for one of the actions
                    # choose action based on ucb scores, added with dirichlet noise (only at root node) for exploration - represent tree policy of mcts
                    action = self.get_noisyUCB_action(sk)

                    # get eps greedy action based on visit counts of child nodes
                    # action = self.get_epsgreedy_visitcounts_action(sk)

                    # check if child node exists in the tree for this action
                    if action.item() in sk.children:
                        # take the action, get the reward
                        next_state, reward = self.gnet.forward(sk.state, action)
                        # store the transition in the trajectory container
                        trajectory.append([sk, action, reward])
                        # step to child node
                        assert(torch.equal(next_state, sk.children[action.item()].state)) # just a sanity check
                        sk = sk.children[action.item()]
                    else:
                        # add new (child) node to the tree
                        reward, bootstrap_value = self.add_node(sk, action)
                        # store the transition in the trajectory container
                        trajectory.append([sk, action, reward])
                        # end of this planning iter - backup estimates and start new planning iter from root node
                        self.backup_estimates(trajectory, bootstrap_value)
                        plan_iter_done = True
                else:
                    # expand node by sampling action from the simulation policy
                    # action = np.random.choice(self.num_actions, p=sk.prior_p)
                    idx = sk.prior_p.multinomial(num_samples=1, replacement=True)
                    idx = idx.squeeze()
                    action = torch.arange(self.num_actions)[idx]
                    sk.expanded = True # set current node as expanded
                    # add new (child) node to the tree
                    reward, bootstrap_value = self.add_node(sk, action)
                    # store the transition in the trajectory container
                    trajectory.append([sk, action, reward])
                    # end of this planning iter - backup estimates and start new planning iter from root node
                    self.backup_estimates(trajectory, bootstrap_value)
                    plan_iter_done = True

        # done with all planning iters
        # return pi_stats of the root node - based on visit counts of child nodes
        pi_stats, value = self.get_pi_stats_and_value()
        return pi_stats, value



# function to get epsilon greedy action
def get_epsgreedy_action(best_action, num_actions, eps):
    action = best_action
    random_prob = random.uniform(0,1)
    if random_prob < eps:
        action = np.random.choice(num_actions)
    return action


# function to conduct training in muzero by sampling trajectories from replay_buffer
def muzero_train(replay_buffer):
    # sample a random trajectory from replay buffer
    idx = torch.randint(len(replay_buffer), size=(1,))
    idx = idx.squeeze()
    trajectory = replay_buffer[idx]
    # select random timestep in the trajectory to start the rollout from
    t = np.random.choice(len(trajectory))
    # select number of rollout steps
    rollout_steps = np.random.choice(len(trajectory) - t)

    # get start state
    hnet.reset()
    obs_seq = []
    for i in range(t+1):
        obs_seq.append(trajectory[i][0])
    s0 = hnet.forward(obs_seq)

    # rollout
    sk = s0
    total_loss = 0
    total_reward_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    for k in range(rollout_steps+1):
        v,p = fnet.forward(sk)
        action, target_reward = trajectory[t+k][1], trajectory[t+k][2]
        action = torch.tensor(action)
        sk_next, reward = gnet.forward(sk, action)

        # reward loss
        reward_loss = (reward - target_reward) ** 2

        # policy loss
        target_p = trajectory[t+k][3]
        cel = nn.CrossEntropyLoss()
        policy_loss = cel(p.unsqueeze(0), target_p.unsqueeze(0))

        # value loss
        bootstrap_steps = np.random.choice(len(trajectory) - (t+k)) # n for n-step bootstrap
        target_v = 0
        for n in range(bootstrap_steps):
            target_v += (df ** n) * trajectory[t+k+n][2]
        target_v += (df ** bootstrap_steps) * trajectory[t+k+bootstrap_steps][4]
        value_loss = (v - target_v) ** 2

        # total loss
        total_loss +=  reward_loss + policy_loss + value_loss
        total_reward_loss += reward_loss
        total_policy_loss += policy_loss
        total_value_loss += value_loss

        # for next step in the rollout
        sk = sk_next

    #rollout done with losses collected for each step of the rollout
    # train op
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_reward_loss, total_policy_loss, total_value_loss





# load environment
env = gym.envs.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

# init functions
hnet = H_net(obs_dim, obs_dim) # keeping h_dim of lstm = obs_dim, therefore abstract_state_dim = obs_dim
gnet = G_net(obs_dim, obs_dim)
fnet = F_net(obs_dim, obs_dim, num_actions)

# hyperparams
num_episodes = 1000
lr = 1e-3
df = 0.997
dirichlet_alpha = 0.3
eps = 1 # decayed
eps_decay_step = 0.01
planning_iterations = 25
training_iterations = 5

# optimizer
params = list(hnet.parameters()) + list(gnet.parameters()) + list(fnet.parameters())
optimizer = torch.optim.Adam(params, lr=lr)

# run
ep_avg_return = [0] # moving average of episode return - used to gauge progress
ep_returns = []
ep_reward_loss = []
ep_policy_loss = []
ep_value_loss = []
replay_buffer = [] # to store (last 100) episode trajectory used for training
rep_buf_size = 10

for ep in range(1, num_episodes):
    # decay eps
    eps_decay_epoch = num_episodes * eps_decay_step
    if ep % eps_decay_epoch == 0:
        eps -= eps_decay_step
        # eps -= 0

    # reset env for new episode
    obs = env.reset()
    hnet.reset()
    ep_return = 0
    ep_length = 0
    ep_trajectory = [] # stores (observation, action, reward, pi_stats, value) tuples constituting the episode trajectory
    ep_obs_seq = [obs] # sequence of observations in the episode - used for hnet

    # MuZero - episode
    while True:
        # if ep % 100 == 0:
        #     env.render()

        with torch.no_grad():

            # get pi stats and value for the current observation (state) from MuZero - MCTS
            mcts_object = MCTS(ep_obs_seq, hnet, fnet, gnet, num_actions, eps, df, dirichlet_alpha, planning_iterations)
            pi_stats, value = mcts_object.mcts()

            # action according to pi_stats
            # best_action = np.random.choice(num_actions, p=pi_stats)
            best_action = torch.argmax(pi_stats).item()

            # get eps-greedy action - to promote some exploration
            # todo - use temperature instead
            action = get_epsgreedy_action(best_action, num_actions, eps)

            # take action
            next_obs, reward, done, _ = env.step(action)

            # update episode return and store transition
            ep_return += (df ** ep_length) * reward
            ep_trajectory.append([obs, action, reward, pi_stats, value])
            ep_length += 1
            ep_obs_seq.append(obs)

            # for next step in episode
            obs = next_obs
            if done:
                if ep % 100 == 0:
                    print('ep:{} \t  ep_return:{}'.format(ep, ep_return))

                # updadte episode avg return
                ep_returns.append(ep_return)
                prev_avg_return = ep_avg_return[-1]
                n = len(ep_avg_return) + 1
                new_avg_return = ((n-1) * prev_avg_return + ep_return) / n
                ep_avg_return.append(new_avg_return)

                # append trajectory to replay buffer
                replay_buffer.append(ep_trajectory)
                if len(replay_buffer) > rep_buf_size:
                    replay_buffer = replay_buffer[-rep_buf_size:]
                break

    # MuZero - train by unrolling sampled trajectories from replay_buffer
    mean_reward_loss = 0
    mean_policy_loss = 0
    mean_value_loss = 0
    for it in range(training_iterations):
        reward_loss, policy_loss, value_loss = muzero_train(replay_buffer)
        mean_reward_loss += (reward_loss - mean_reward_loss) / (it+1)
        mean_policy_loss += (policy_loss - mean_policy_loss) / (it+1)
        mean_value_loss += (value_loss - mean_value_loss) / (it+1)
    ep_reward_loss.append(mean_reward_loss.detach().numpy())
    ep_policy_loss.append(mean_policy_loss.detach().numpy())
    ep_value_loss.append(mean_value_loss.detach().numpy())


# plot
fig, ax = plt.subplots(2,2)
ax[0,0].plot(ep_reward_loss, color='green', label='ep_reward_loss')
ax[0,1].plot(ep_policy_loss, color='red', label='ep_policy_loss')
ax[1,0].plot(ep_value_loss, color='blue', label='ep_value_loss')
ax[1,0].set_ylim([0,1000])
ax[1,1].plot(ep_returns, color='black', label='ep_return')
ax[1,1].plot(ep_avg_return[1:], color='yellow', label='ep_avg_return')
plt.legend()
plt.xlabel('episode')
plt.ylabel('ep return')
plt.savefig('cartpole_MuZero_ucb_repBufSize='+str(rep_buf_size)+'_epsDecay='+str(eps_decay_step)+'_lr='+str(lr)+'_planIter='+str(planning_iterations)+'_trainIter='+str(training_iterations)+'_numEP='+str(num_episodes)+'.png')
