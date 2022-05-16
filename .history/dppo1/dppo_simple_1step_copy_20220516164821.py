from re import A
from time import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal
import gym
import os
import numpy as np
import random
import multiprocessing
from copy import deepcopy


torch.manual_seed(0)
def setup_seed(seed):
	"""设置随机数种子函数"""
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True


env_name='Pendulum-v0'
has_continuous_action_space = True
env = gym.make(env_name)

state_dim = env.observation_space.shape[0]
if has_continuous_action_space:
	action_dim = env.action_space.shape[0]
else:
	action_dim = env.action_space.n
env.close()
lr_actor = 0.0003
lr_critic = 0.001
eps_clip = 0.2
gamma = 0.99
lr_actor = 0.0003
lr_critic = 0.001
action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = int(2.5e5)
update_timestep = 2500
K_epochs = 80
action_std_init=0.6
# device = torch.device('cpu')

if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


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
		self.optimizer = torch.optim.Adam([
			{'params': self.actor.parameters(), 'lr': lr_actor},
			{'params': self.critic.parameters(), 'lr': lr_critic}
		])

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


class PPO:
	def __init__(self, policy, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
				 has_continuous_action_space, action_std_init=0.6,device):

		self.has_continuous_action_space = has_continuous_action_space
		self.action_std = action_std_init
		self.gamma = gamma
		self.eps_clip = eps_clip
		self.K_epochs = K_epochs
		self.buffer = RolloutBuffer()
		self.policy = policy
		# self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
		self.optimizer = policy.optimizer

		self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(policy.device)
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
		self.set_action_std(self.action_std)




	def select_action(self, state):
		with torch.no_grad():
			state = torch.FloatTensor(state).to(device)
			action, action_logprob = self.policy_old.act(state)
		self.buffer.states.append(state)
		self.buffer.actions.append(action)
		self.buffer.logprobs.append(action_logprob)
		return action.detach().cpu().numpy().flatten()



	def update(self,buffer):
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
		# value_previous=next_value
		rets_temp=[]
		advs_temp=[]
		value_previous = buffer.vals[-1]
		for rew, don, val in zip(reversed(buffer.rewards),reversed(buffer.dones),reversed(buffer.vals[:-1])):
			ret = self.gamma * value_previous * (1 - don) + rew
			rets_temp.append(ret)
			# delta = rew + value_previous * self.gamma * (1 - don) - val
			# adv = delta + (1 - don) * adv * self.gamma * 0.95   #self.lam=0.95

			advs_temp.append(ret-val)
			value_previous=val
		buffer.rets=list(reversed(rets_temp))
		buffer.advs=list(reversed(advs_temp))

		advs=torch.squeeze(torch.stack(buffer.advs, dim=0)).detach().to(device)
		advs=(advs-advs.mean())/(advs.std()+1e-7)
		rets=torch.squeeze(torch.stack(buffer.rets, dim=0)).detach().to(device)
		# convert list to tensor
		states = torch.squeeze(torch.stack(buffer.states, dim=0)).detach().to(device)
		actions = torch.squeeze(torch.stack(buffer.actions, dim=0)).detach().to(device)
		old_logprobs = torch.squeeze(torch.stack(buffer.logprobs, dim=0)).detach().to(device)


		for _ in range(self.K_epochs):
			# Evaluating old actions and values
			logprobs, values, dist_entropy = self.policy.evaluate(states, actions)
			values = torch.squeeze(values)

			# Finding the ratio (pi_theta / pi_theta__old)
			ratios = torch.exp(logprobs - old_logprobs.detach())

			#advantages = rewards - values.detach()
			# print(len(ratios),len(advs))
			surr1 = ratios * advs
			surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advs


			loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(values, rets) - 0.01 * dist_entropy
			self.optimizer.zero_grad()
			loss.mean().backward()
			self.optimizer.step()
		self.policy_old.load_state_dict(self.policy.state_dict())
		# self.buffer.clear()

	def collect(self, env):
		state = env.reset()
		ep_reward = 0
		self.buffer.clear()
		time_step = 0
		for t in range(1000):
			# env.render()
			action = self.select_action(state)
			state_, reward, done, _ = env.step(action)

			state = torch.FloatTensor(state).to(device)
			val=self.policy.critic(state)      #self_add

			self.buffer.vals.append(val.detach())       #self_add
			self.buffer.rewards.append(reward)
			self.buffer.dones.append(done)

			time_step +=1
			ep_reward += reward
			# if time_step % update_timestep == 0:    #update_timestep=2500
			#     state_ = torch.FloatTensor(state_).to(device)
			#     val_next=self.policy.critic(state_)      #self_add
			#     self.buffer.vals.append(val_next.detach())
				# self.update(val_next)


			if done:
				state_ = torch.FloatTensor(state_).to(device)
				val_next=self.policy.critic(state_)      #self_add
				self.buffer.vals.append(val_next.detach())
				break
			state=state_
		return ep_reward, time_step
		

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
	K_epochs = 80
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


	net = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
	ppo_agent = PPO(deepcopy(net), state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

	process_num = 3#6
	pipe_dict = dict((i, (pipe1, pipe2)) for i in range(process_num)
					 for pipe1, pipe2 in (multiprocessing.Pipe(),))
	child_process_list = []
	for i in range(process_num):
		pro = multiprocessing.Process(
			target=child_process2, args=(pipe_dict[i][1],))
		child_process_list.append(pro)
	[pipe_dict[i][0].send(net) for i in range(process_num)]
	[p.start() for p in child_process_list]

	directory = "PPO_preTrained1"
	if not os.path.exists(directory):
		os.makedirs(directory)
	directory = directory + '/'
	if not os.path.exists(directory):
		os.makedirs(directory)

	time_step = 0
	rewardList = []
	buffer_list = []
	for i_episode in range(2000):
		# state = env.reset()
		ep_reward = 0
		# ep_reward = ppo_agent.collect(env)
		for i in range(process_num):
			receive = pipe_dict[i][0].recv()
			data = receive[0]
			buffer_list.append(data)
			ep_reward += receive[1]
			time_step += receive[2]

		print(i_episode, ep_reward/process_num)
		rewardList.append(ep_reward)
		reward_list = [i_episode, ep_reward]
		for buffer in buffer_list:
			ppo_agent.update(buffer)
			
		if has_continuous_action_space and time_step % action_std_decay_freq == 0:
			ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

		net.load_state_dict(ppo_agent.policy.state_dict())
		[pipe_dict[i][0].send(net) for i in range(process_num)]
		# reward_file_name = 'ppo_simple_2.txt'
		# with open(reward_file_name, 'a') as reward_txt:
		# 	reward_txt.write(str(reward_list) + '\n')
		i_episode += 1
	env.close()

def child_process2(pipe):
	setup_seed(0)
	env_name='Pendulum-v0'
	env = gym.make(env_name)
	#env = ArmEnv(mode='hard')

	env.reset()
	while True:
		#env.render()
		net = pipe.recv()  # 收主线程的net参数，这句也有同步的功能
		ppo_agent = PPO(net, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

		rewards,time_step = ppo_agent.collect(env)
		transition = ppo_agent.buffer
		buffer = RolloutBuffer()
		buffer.rewards = transition.rewards
		buffer.dones = transition.dones
		buffer.actions = transition.actions
		buffer.states = transition.states
		buffer.logprobs = transition.logprobs
		buffer.vals = transition.vals
		# r = transition.rewards
		# m = transition.dones
		# a = transition.actions
		# s = transition.states
		# log = transition.logprobs
		# data = (r, m, s, a, log)
		"""pipe不能直接传输buffer回主进程，可能是buffer内有transition，因此将数据取出来打包回传"""
		pipe.send((transition, rewards,time_step))



if __name__ == '__main__':
	train()





