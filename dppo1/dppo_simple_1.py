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
max_ep_len = 1000
device = torch.device('cpu')

# if (torch.cuda.is_available()):
#     device = torch.device('cuda:0')
#     torch.cuda.empty_cache()
#     print("Device set to : " + str(torch.cuda.get_device_name(device)))
# else:
#     print("Device set to : cpu")


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
	
	def insert(self, data):
		self.states += data.states
		self.actions += data.actions
		self.rewards += data.rewards
		self.dones += data.dones
		self.vals += data.vals
		self.logprobs += data.logprobs
		self.rets += data.rets
		self.advs += data.advs



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
	def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init,device='cpu'):
		super(ActorCritic, self).__init__()
		self.device = device 
		self.has_continuous_action_space = has_continuous_action_space
		self.action_dim = action_dim
		self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
		self.actor=MLPActor(state_dim, action_dim)
		self.critic=MLPQFunction(state_dim)
		self.optimizer = torch.optim.Adam([
			{'params': self.actor.parameters(), 'lr': lr_actor},
			{'params': self.critic.parameters(), 'lr': lr_critic}
		])
		# self.step = 0

	def set_action_std(self, new_action_std):
		self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)

	def act(self, state, evaluate=False):
		# self.step += 1
		action_mean = self.actor(state)
		cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
		dist = MultivariateNormal(action_mean, cov_mat)
		if evaluate:
			return np.clip(dist.mean.detach().cpu().numpy().flatten(),-2,2)
		action = dist.sample()
		action=np.clip(action,-2,2)
		action_logprob = dist.log_prob(action)
		return action.detach(), action_logprob.detach()

	def evaluate(self, state, action):
		action_mean = self.actor(state)
		action_var = self.action_var.expand_as(action_mean)
		cov_mat = torch.diag_embed(action_var).to(self.device)
		dist = MultivariateNormal(action_mean, cov_mat)
		if self.action_dim==1:
			action=action.reshape(-1,self.action_dim)
		action_logprobs = dist.log_prob(action)
		dist_entropy = dist.entropy()
		state_values = self.critic(state)
		return action_logprobs, state_values, dist_entropy


class PPO:
	def __init__(self, policy, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
				 has_continuous_action_space, action_std_init=0.6, device='cpu'):

		self.has_continuous_action_space = has_continuous_action_space
		self.action_std = action_std_init
		self.gamma = gamma
		self.eps_clip = eps_clip
		self.K_epochs = K_epochs
		self.device = device
		self.buffer = RolloutBuffer()
		self.policy = policy
		# self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
		self.optimizer = policy.optimizer

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

		discounted_reward = 0
		rets_temp=[]
		advs_temp=[]
		for reward, is_terminal in zip(reversed(buffer.rewards), reversed(buffer.dones)):
			if is_terminal:
				discounted_reward = 0
			discounted_reward = reward + (self.gamma * discounted_reward)
			rets_temp.append(discounted_reward)
		# value_previous=next_value
		"""
		value_previous = buffer.vals[-1]
		for rew, don, val in zip(reversed(buffer.rewards),reversed(buffer.dones),reversed(buffer.vals[:-1])):
			ret = self.gamma * value_previous * (1 - don) + rew
			rets_temp.append(ret)
			# delta = rew + value_previous * self.gamma * (1 - don) - val
			# adv = delta + (1 - don) * adv * self.gamma * 0.95   #self.lam=0.95

			advs_temp.append(ret-val)
			value_previous=val
		"""
		buffer.rets=list(reversed(rets_temp))
		# buffer.advs=list(reversed(advs_temp))

		# advs=torch.squeeze(torch.stack(buffer.advs, dim=0)).detach().to(device)
		# advs=(advs-advs.mean())/(advs.std()+1e-7)
		rets = torch.tensor(buffer.rets, dtype=torch.float32).to(device)
		# rets=torch.squeeze(torch.stack(buffer.rets, dim=0)).detach().to(device)

		rets = (rets - rets.mean()) / (rets.std() + 1e-7)
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

			advs = rets - values.detach()
			surr1 = ratios * advs
			surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advs
			# print(surr1.shape, self.MseLoss(values,rets).shape)


			loss = -torch.min(surr1, surr2).sum(dim=-1) + 0.5 * self.MseLoss(values, rets) - 0.01 * dist_entropy
			self.optimizer.zero_grad()
			loss.mean().backward()
			self.optimizer.step()
		# print(self.MseLoss(values, rets))
		self.policy_old.load_state_dict(self.policy.state_dict())
		# self.buffer.clear()
	def evaluate(self,env_eval):
		# env_eval = gym.make(env_name)
		ep_rewards = []
		for _ in range(10):
			ep_reward = 0
			state = env_eval.reset()
			for t in range(max_ep_len):
				# env.render()
				# with torch.no_grad():
					# print(state)
				# action = self.select_action(state)
				# with torch.no_grad():
				state = torch.FloatTensor(state).to(device)
				action = self.policy_old.act(state, evaluate=True)
				state_, reward, done, _ = env_eval.step(action)
				ep_reward += reward
				if done or t==max_ep_len-1:
					break
				state=state_
			ep_rewards.append(ep_reward)
		return np.mean(ep_rewards)

	def collect(self, env):
		state = env.reset()
		ep_reward = 0
		self.buffer.clear()
		time_step = 0
		for t in range(max_ep_len):
			# env.render()
			action = self.select_action(state)
			state_, reward, done, _ = env.step(action)
			# print(reward,done)

			state = torch.FloatTensor(state).to(device)
			val=self.policy.critic(state)      #self_add

			self.buffer.vals.append(val.detach())       #self_add
			self.buffer.rewards.append(reward)
			self.buffer.dones.append(done)

			time_step += 1
			ep_reward += reward
			# if time_step % update_timestep == 0:    #update_timestep=2500
			#     state_ = torch.FloatTensor(state_).to(device)
			#     val_next=self.policy.critic(state_)      #self_add
			#     self.buffer.vals.append(val_next.detach())
				# self.update(val_next)


			if done or t==max_ep_len-1:
				# state_ = torch.FloatTensor(state_).to(device)
				# val_next = self.policy.critic(state_)      #self_add
				# self.buffer.vals.append(val_next.detach())
				break
			state=state_
		return ep_reward, time_step
		

	def save(self, checkpoint_path):
		torch.save(self.policy_old.state_dict(), checkpoint_path)

	def load(self, checkpoint_path):
		self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
		self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


def train():
	has_continuous_action_space = True
	action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
	action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
	min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
	action_std_decay_freq = int(2.5e5)
	update_timestep = 2500
	# K_epochs = 10
	# eps_clip = 0.2
	# gamma = 0.99
	# lr_actor = 0.0003
	# lr_critic = 0.001

	print("training environment name : " + env_name)
	env = gym.make(env_name)
	env.seed(0)

	state_dim = env.observation_space.shape[0]
	if has_continuous_action_space:
		action_dim = env.action_space.shape[0]
	else:
		action_dim = env.action_space.n


	net = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
	ppo_agent = PPO(deepcopy(net), state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=action_std)

	process_num = 3#6
	pipe_dict = dict((i, (pipe1, pipe2)) for i in range(process_num)
					 for pipe1, pipe2 in (multiprocessing.Pipe(),))
	child_process_list = []
	for i in range(process_num):
		pro = multiprocessing.Process(
			target=child_process2, args=(pipe_dict[i][1],))
		child_process_list.append(pro)
	[pipe_dict[i][0].send((deepcopy(ppo_agent.policy_old),ppo_agent.action_std)) for i in range(process_num)]
	[p.start() for p in child_process_list]

	directory = "PPO_preTrained1"
	if not os.path.exists(directory):
		os.makedirs(directory)
	directory = directory + '/'
	if not os.path.exists(directory):
		os.makedirs(directory)

	time_step = 0
	buffer_all = RolloutBuffer()
	episode_reward = []
	for i_episode in range(2000):
		# state = env.reset()
		ep_reward = 0
		# ep_reward = ppo_agent.collect(env)
		for i in range(process_num):
			receive = pipe_dict[i][0].recv()
			data = receive[0]
			buffer_all.insert(data)
			ep_reward += receive[1]
			time_step += receive[2]

		print(i_episode, ep_reward/process_num)
		reward_list = [i_episode, ep_reward]
		if (i_episode+1) % 4 == 0:
			ppo_agent.update(buffer_all)
		# if time_step % update_timestep == 0:
			# ppo_agent.update(buffer_all)
			buffer_all.clear()
			
		if has_continuous_action_space and i_episode*1250 % action_std_decay_freq == 0:
			ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)
			
		if i_episode % 5 == 0:
			reward = ppo_agent.evaluate(env)
			print(f'episode {i_episode} evaluate reward:',reward)
		# net.action_var = ppo_agent.policy.action_var
		# net.load_state_dict(ppo_agent.policy.state_dict())
		# print(net.action_var)
		[pipe_dict[i][0].send((deepcopy(ppo_agent.policy_old),ppo_agent.action_std)) for i in range(process_num)]
		reward_file_name = 'dppo_simple_3.txt'
		with open(reward_file_name, 'a') as reward_txt:
			reward_txt.write(str(reward_list) + '\n')
		i_episode += 1
		episode_reward.append(ep_reward/process_num)
		np.save('dppo_3.npy', episode_reward)
	env.close()
	[p.terminate() for p in child_process_list]


def child_process2(pipe):
	setup_seed(0)
	env = gym.make(env_name)
	#env = ArmEnv(mode='hard')

	env.reset()
	while True:
		#env.render()
		net, cur_action_std = pipe.recv()  # 收主线程的net参数，这句也有同步的功能
		ppo_agent = PPO(net, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=cur_action_std)

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
		pipe.send((buffer, rewards, time_step))
		# ppo_agent.buffer.clear()



if __name__ == '__main__':
	train()





