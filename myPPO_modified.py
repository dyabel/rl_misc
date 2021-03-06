#!/usr/bin/python
# -*- coding: utf-8 -*-
from py import process
from PPOModel import *
import gym
import multiprocessing
from copy import deepcopy
import time
import random


def setup_seed(seed):
	"""设置随机数种子函数"""
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True


def main2():
	"""
	第二种实现多进程训练的思路：
		1.多个进程不训练网络，只是拿到主进程的网络后去探索环境，并将transition通过pipe传回主进程
		2.主进程将所有子进程的transition打包为一个buffer后供网络训练
		3.将更新后的net再传到子进程，回到1
	"""

	env = gym.make('LunarLanderContinuous-v2')
	env.seed(0)
	net = GlobalNet(env.observation_space.shape[0], env.action_space.shape[0])


	ppo = AgentPPO(deepcopy(net))
	process_num = 3#6
	pipe_dict = dict((i, (pipe1, pipe2)) for i in range(process_num)
					 for pipe1, pipe2 in (multiprocessing.Pipe(),))
	child_process_list = []
	for i in range(process_num):
		pro = multiprocessing.Process(
			target=child_process2, args=(pipe_dict[i][1],))
		child_process_list.append(pro)
	[pipe_dict[i][0].send(net) for i in range(process_num)]
	# # [p.start() for p in child_process_list]
	# time.sleep(10)
	for i in range(process_num):
		pipe_dict[i][0].send(net)
		# time.sleep(1)
	for p in child_process_list:
		p.start()
		# time.sleep(1)
	[p.join() for p in child_process_list]

	rewardList = list()
	timeList = list()
	begin = time.time()
	MAX_EPISODE = 150
	batch_size = 128
	max_reward = -np.inf
	for episode in range(MAX_EPISODE):
		print(episode)
		reward = 0
		buffer_list = list()
		for i in range(process_num):
			# 这句带同步子进程的功能，收不到子进程的数据就都不会走到for之后的语句
			time.sleep(2)
			print(111)
			receive = pipe_dict[i][0].recv()
			data = receive[0]
			buffer_list.append(data)
			reward += receive[1]
		ppo.update_policy_mp(batch_size, 8, buffer_list)    #训练网络
		net.act.load_state_dict(ppo.act.state_dict())
		net.cri.load_state_dict(ppo.cri.state_dict())
		for i in range(process_num):
			time.sleep(1)
			pipe_dict[i][0].send(net)
		# [pipe_dict[i][0].send(net) for i in range(process_num)]
		reward /= process_num
		rewardList.append(reward)
		timeList.append(time.time() - begin)
		print(f'episode:{episode}  reward:{reward} time:{timeList[-1]}')

		reward_list = [episode, reward]
		reward_file_name = '3-process.txt'
		with open(reward_file_name, 'a') as reward_txt:
			reward_txt.write(str(reward_list) + '\n')
		#if reward > max_reward and episode > MAX_EPISODE*2/3:
		#    max_reward = reward
			#torch.save(net.act.state_dict(), '../TrainedModel/act.pkl')    #自己注释

	[p.terminate() for p in child_process_list]


def child_process2(pipe):
	setup_seed(0)
	env = gym.make('LunarLanderContinuous-v2')
	#env = ArmEnv(mode='hard')

	env.reset()
	while True:
		#env.render()
		print('#'*100)
		time.sleep(1)
		net = pipe.recv()  # 收主线程的net参数，这句也有同步的功能
		print('*'*100)
		ppo = AgentPPO(net, if_explore=True)
		rewards, steps = ppo.update_buffer(env, 500, 1)
		transition = ppo.buffer.sample_all()
		r = transition.reward
		m = transition.mask
		a = transition.action
		s = transition.state
		log = transition.log_prob
		data = (r, m, s, a, log)
		"""pipe不能直接传输buffer回主进程，可能是buffer内有transition，因此将数据取出来打包回传"""
		pipe.send((data, rewards))


if __name__ == "__main__":
	main2()
