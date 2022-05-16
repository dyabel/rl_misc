import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal
import gym
import os
import numpy as np


torch.manual_seed(0)




device = torch.device('cpu')
"""
if (torch.cuda.is_available()):
    device = torch.device('cuda:7')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
"""
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias

class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        # print(action_mean.shape,action_logstd.shape)
        return FixedNormal(action_mean, action_logstd.exp())

class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []
        self.logprobs = []
        self.rets = []
        self.advs = []


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
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
        self.fc1 = init_(nn.Linear(obs_dim,64))
        self.fc2 = init_(nn.Linear(64,64))
        # self.fc3=nn.Linear(64, 64)
        # self.fc4=nn.Linear(64,act_dim)
        self.act_layer = DiagGaussian(64, act_dim)

    def forward(self, x):
        x=torch.tanh(self.fc1(x))
        x=torch.tanh(self.fc2(x))
        # x=torch.tanh(self.fc3(x))

        return self.act_layer(x)

class MLPQFunction(nn.Module):
    def __init__(self, obs_dim):   #act_dim=1.0
        super().__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
        self.fc1 = init_(nn.Linear(obs_dim, 64))
        self.fc2 = init_(nn.Linear(64, 64))
        # self.fc3 = nn.Linear(64, 64)
        self.fc4 = init_(nn.Linear(64,1))

    def forward(self, obs):
        x=torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))
        # x=torch.tanh(self.fc3(x))
        q=self.fc4(x)
        return q


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space
        self.action_dim = action_dim
        # self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        self.actor=MLPActor(state_dim, action_dim)
        self.critic=MLPQFunction(state_dim)

    # def set_action_std(self, new_action_std):
        # self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def act(self, state):
        if len(state.shape) == 1: 
            state = state.unsqueeze(0)
        dist = self.actor(state)
        # cov_mat = torch.exp(self.action_var).unsqueeze(dim=0)
        # cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        # dist = FixedNormal(action_mean.cpu(), cov_mat.cpu())
        # dist = MultivariateNormal(action_mean.cpu(), cov_mat.cpu())
        action = dist.sample()
        action=np.clip(action,-2,2)
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        if len(state.shape) == 1: 
            state = state.unsqueeze(0)
        dist = self.actor(state)
        # action_var = self.action_var.expand_as(action_mean)
        # cov_mat = torch.diag_embed(action_var).to(device)
        # dist = MultivariateNormal(action_mean, cov_mat)
        if self.action_dim==1:
            action=action.reshape(-1,self.action_dim)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space
        self.action_std = action_std_init
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
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
        # self.set_action_std(self.action_std)




    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        return action.detach().cpu().numpy().flatten()



    def update(self,next_value):
        # Monte Carlo estimate of returns
        '''
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.buffer.rewards), reversed(self.buffer.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        '''
        ret=0
        adv=0
        value_previous = next_value
        rets_temp=[]
        advs_temp=[]
        # print(len(self.buffer.vals),len(self.buffer.rewards),len(self.buffer.dones))
        for rew, don, val in zip(reversed(self.buffer.rewards),reversed(self.buffer.dones),reversed(self.buffer.vals)):
            # ret = self.gamma * ret * (1 - don) + val
            # rets_temp.append(ret)
            delta = rew + value_previous * self.gamma * (1 - don) - val
            adv = delta + (1 - don) * adv * self.gamma * 0.95   #self.lam=0.95

            advs_temp.append(adv)
            rets_temp.append(adv + val)
            value_previous=val
        self.buffer.rets=list(reversed(rets_temp))
        self.buffer.advs=list(reversed(advs_temp))

        advs=torch.squeeze(torch.stack(self.buffer.advs, dim=0)).detach().to(device)
        advs=(advs-advs.mean())/(advs.std()+1e-7)
        rets=torch.squeeze(torch.stack(self.buffer.rets, dim=0)).detach().to(device)
        # convert list to tensor
        states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)


        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, values, dist_entropy = self.policy.evaluate(states, actions)
            values = torch.squeeze(values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs.squeeze() - old_logprobs.detach())

            #advantages = rewards - values.detach()
            surr1 = ratios * advs
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advs

            # print(surr1.shape,values.shape,dist_entropy.shape,ratios.shape, advs.shape,logprobs.shape,old_logprobs.shape)
            loss = -torch.min(surr1, surr2).mean() + 0.5 * self.MseLoss(values, rets) - 0.01 * dist_entropy.mean()
            # print(values,rets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # print(self.MseLoss(values, rets),rets)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


def train():
    env_name='Pendulum-v0'
    has_continuous_action_space = True
    max_ep_len = 1000
    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)
    update_timestep = 2500
    K_epochs = 10
    eps_clip = 0.2
    gamma = 0.99
    lr_actor = 0.0003
    lr_critic = 0.001

    print("training environment name : " + env_name)
    env = gym.make(env_name)
    env.seed(0)

    state_dim = env.observation_space.shape[0]
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n


    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    directory = "PPO_preTrained1"
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = directory + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    time_step = 0
    rewardList = []

    for i_episode in range(2000):
        state = env.reset()
        ep_reward = 0

        for t in range(1000):
            # env.render()
            action = ppo_agent.select_action(state)
            state_, reward, done, _ = env.step(action)

            state = torch.FloatTensor(state).to(device)
            val=ppo_agent.policy.critic(state)      #self_add

            ppo_agent.buffer.vals.append(val)       #self_add
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.dones.append(done)

            time_step +=1
            ep_reward += reward

            # if has_continuous_action_space and time_step % action_std_decay_freq == 0:
            #     ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            
            if time_step % update_timestep == 0 or done:    #update_timestep=2500
                state_ = torch.FloatTensor(state_).to(device)
                next_val=ppo_agent.policy.critic(state_)      #self_add
                ppo_agent.update(next_val)
                time_step = 0
            state=state_

        print(i_episode, ep_reward)
        rewardList.append(ep_reward)
        reward_list = [i_episode, ep_reward]
        reward_file_name = 'ppo_simple_2.txt'
        with open(reward_file_name, 'a') as reward_txt:
            reward_txt.write(str(reward_list) + '\n')
        i_episode += 1
    env.close()




if __name__ == '__main__':
    train()





