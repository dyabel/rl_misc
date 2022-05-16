from .ppo_simple_1step import *
import torch.multiprocessing as mp
from chief import chief_worker  

num_of_workers = mp.cpu_count() - 1
processor = []
workers = []

p = mp.Process(target=chief_worker, args=(num_of_workers, traffic_signal, critic_counter, actor_counter, 
	critic_shared_model, actor_shared_model, critic_shared_grad_buffer, actor_shared_grad_buffer, 
	critic_optimizer, actor_optimizer, shared_reward, shared_obs_state, args.policy_update_step, args.env_name))

processor.append(p)


for idx in range(num_of_workers):
	workers.append(dppo_agent.dppo_workers(args))

for worker in workers:
	p = mp.Process(target=worker.train_network, args=(traffic_signal, critic_counter, actor_counter, 
		critic_shared_model, actor_shared_model, shared_obs_state, critic_shared_grad_buffer, actor_shared_grad_buffer, shared_reward))
	processor.append(p)

for p in processor:
	p.start()

for p in processor:
	p.join()