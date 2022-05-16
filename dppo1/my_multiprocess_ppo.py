import gym
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import multiprocessing
from copy import deepcopy
import random
env='Pendulum-v0'
state_dim=env.observation_space.shape[0]
action_dim=env.action_space.shape[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []      #
        self.logprobs = []
        self.rets = []      #
        self.advs = []      #


    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.dones[:]
        del self.vals[:]
        del self.logprobs[:]
        del self.rets[:]
        del self.advs[:]



class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1=nn.Linear(obs_dim,64)
        self.fc2=nn.Linear(64,64)
        self.fc3=nn.Linear(64, 64)
        self.fc4=nn.Linear(64,act_dim)

    def forward(self, x):
        x=torch.tanh(self.fc1(x))
        x=torch.tanh(self.fc2(x))
        x=torch.tanh(self.fc3(x))
        pi=torch.tanh(self.fc4(x))
        return pi

class MLPQFunction(nn.Module):
    def __init__(self, obs_dim):   #act_dim=1.0
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4=nn.Linear(64,1)

    def forward(self, obs):
        x=torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))
        x=torch.tanh(self.fc3(x))
        q=self.fc4(x)
        return q

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        self.actor=MLPActor(state_dim, action_dim)
        self.critic=MLPQFunction(state_dim)

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def act(self, state):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action=np.clip(action,-2,2)
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)
        if self.action_dim==1:
            action=action.reshape(-1,self.action_dim)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy


class GlobalNet:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.act_cri = ActorCritic(state_dim, action_dim, True, action_std_init=0.6 )
        self.optimizer = torch.optim.Adam([
            {'params': self.act_ori.actor.parameters(), 'lr': 0.0003},
            {'params': self.act_ori.critic.parameters(), 'lr': 0.001}
        ])
        '''
        self.net_dim = 256
        self.learning_rate = 1e-4
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.act = ActorPPO(state_dim, action_dim, self.net_dim)
        self.act_optimizer = torch.optim.Adam(
            self.act.parameters(), lr=self.learning_rate, )  # betas=(0.5, 0.99))
        self.cri = CriticAdv(state_dim, self.net_dim)
        self.cri_optimizer = torch.optim.Adam(
            self.cri.parameters(), lr=self.learning_rate, )  # betas=(0.5, 0.99))
        '''


class PPO:
    def __init__(self, net):
        self.device = net.device
        self.act_cri = net.act_cri.to(net.device)
        self.act_cri_old = net.act_cri.to(net.device)
        self.optimizer = net.optimizer

        self.has_continuous_action_space = True
        self.action_std = 0.6
        self.buffer = RolloutBuffer()


        self.act_cri_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()


    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)


    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
            print("setting actor output action_std to min_action_std : ", self.action_std)
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        return action.detach().cpu().numpy().flatten()

    def update(self, buffer_total):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        #states.shape:[2500,3]  actions.shape:[2500]

        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, values, dist_entropy = self.policy.evaluate(states, actions)
            values = torch.squeeze(values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - values.detach()
            #print('type(advantages),advantages',type(advantages),advantages)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(values, rewards) - 0.01 * dist_entropy
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def work(self, env):
        state = env.reset()
        ep_reward=0
        action_std = 0.6  # starting std for action distribution (Multivariate Normal)
        action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
        min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
        action_std_decay_freq = int(2.5e5)
        time_step=0
        for t in range(1000):
            #env.render()
            action = self.select_action(state)
            state_, reward, done, _ = env.step(action)

            self.buffer.rewards.append(reward)
            self.buffer.dones.append(done)

            time_step+=1
            ep_reward += reward


            if  time_step % action_std_decay_freq == 0:
                self.decay_action_std(action_std_decay_rate, min_action_std)

            if done:
                break
            state=state_


    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


def main2():
    env = gym.make('LunarLanderContinuous-v2')
    env.seed(0)
    net = GlobalNet(env.observation_space.shape[0], env.action_space.shape[0])

    ppo = PPO(deepcopy(net))
    process_num = 3  # 6
    pipe_dict = dict((i, (pipe1, pipe2)) for i in range(process_num)
                     for pipe1, pipe2 in (multiprocessing.Pipe(),))
    child_process_list = []
    for i in range(process_num):
        pro = multiprocessing.Process(
            target=child_process2, args=(pipe_dict[i][1],))
        child_process_list.append(pro)
    [pipe_dict[i][0].send(net) for i in range(process_num)]
    [p.start() for p in child_process_list]

    rewardList = list()
    MAX_EPISODE = 150
    batch_size = 128

    for episode in range(MAX_EPISODE):
        reward = 0
        buffer_list = list()
        for i in range(process_num):
            # 这句带同步子进程的功能，收不到子进程的数据就都不会走到for之后的语句
            receive = pipe_dict[i][0].recv()
            data = receive
            buffer_list.append(data)

        ppo.update_policy_mp(batch_size, 8, buffer_list)  # 训练网络    **************************
        net.act.load_state_dict(ppo.act.state_dict())
        net.cri.load_state_dict(ppo.cri.state_dict())
        [pipe_dict[i][0].send(net) for i in range(process_num)]

        reward /= process_num
        rewardList.append(reward)
        print('episode:',episode,'reward:',reward)
        reward_list = [episode, reward]
        reward_file_name = '3-process.txt'
        with open(reward_file_name, 'a') as reward_txt:
            reward_txt.write(str(reward_list) + '\n')

def child_process2(pipe):
    torch.seed(0)
    env = gym.make('LunarLanderContinuous-v2')

    env.reset()
    while True:
        #env.render()
        net = pipe.recv()  # 收主线程的net参数，这句也有同步的功能
        ppo = PPO(net)
        ppo.work(env)   #与环境交互，采样

        state = ppo.buffer.states
        action = ppo.buffer.actions
        reward = ppo.buffer.rewards
        logprob = ppo.buffer.logprobs
        done = ppo.buffer.dones
        data=[state, action, reward, logprob, done]
        pipe.send(data)




if __name__ == "__main__":
    main2()