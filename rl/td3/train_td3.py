import torch
torch.multiprocessing.set_start_method('forkserver', force=True) # critical for make multiprocessing work
import time
import queue
import math
import random
import datetime
import os
import numpy as np
import torch.multiprocessing as mp
from torch.multiprocessing import Process

from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager

from rl.buffers import ReplayBuffer
from utils.common_func import rand_params
from rl.td3.td3 import TD3_Trainer, worker, cpu_worker


def train_td3(env, cfg):
    basic_cfg, train_cfg, env_cfg = cfg.basic, cfg.train, cfg.env
    seed = basic_cfg.seed
    torch.manual_seed(seed)  # Reproducibility
    np.random.seed(seed)
    random.seed(seed)

    num_workers = basic_cfg.process # or: mp.cpu_count()
    prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model_path = './data/weights/{}{}'.format(prefix, seed)
    if not os.path.exists(model_path) and basic_cfg.train:
        os.makedirs(model_path)
    print('Model Path: ', model_path)
    
    # mujoco env gives the range of action and it is symmetric
    action_range = train_cfg.action_range if train_cfg.action_range else env.action_space.high[0]
        
    # the replay buffer is a class, have to use torch manager to make it a proxy for sharing across processes
    BaseManager.register('ReplayBuffer', ReplayBuffer)
    manager = BaseManager()
    manager.start()
    replay_buffer = manager.ReplayBuffer(train_cfg.replay_buffer_size)  # share the replay buffer through manager

    action_space = env.action_space
    state_space = env.observation_space

    if not torch.cuda.is_available():
        raise NotImplemented

    machine_type = 'gpu' if torch.cuda.is_available() else 'cpu'
    worker_ = worker if torch.cuda.is_available() else cpu_worker
    print(machine_type, worker)
    td3_trainer=TD3_Trainer(replay_buffer, state_space, action_space, train_cfg.hidden_dim, train_cfg.q_lr, train_cfg.policy_lr,\
        policy_target_update_interval=train_cfg.policy_target_update_interval, action_range=action_range, machine_type=machine_type )

    if basic_cfg.train: 
        if basic_cfg.finetune:
            td3_trainer.load_model('./data/weights/'+ basic_cfg.path +'/{}_td3'.format(basic_cfg.model_id))
        td3_trainer.share_memory()

        rewards_queue=mp.Queue()  # used for get rewards from all processes and plot the curve
        eval_rewards_queue = mp.Queue()  # used for get offline evaluated rewards from all processes and plot the curve
        success_queue = mp.Queue()  # used for get success events from all processes
        eval_success_queue = mp.Queue()

        processes=[]
        rewards=[]
        success = []
        eval_rewards = []
        eval_success = []

        for i in range(num_workers):
            process = Process(target=worker_, args=(i+5, td3_trainer, basic_cfg.env_name, env_cfg, rewards_queue, eval_rewards_queue, success_queue, eval_success_queue, \
            train_cfg.eval_interval, replay_buffer, train_cfg.max_episodes, train_cfg.max_steps, train_cfg.batch_size, train_cfg.explore_steps,
            train_cfg.noise_decay, train_cfg.update_itr, train_cfg.explore_noise_scale, train_cfg.eval_noise_scale, \
            train_cfg.reward_scale, train_cfg.gamma, train_cfg.soft_tau, train_cfg.deterministic, train_cfg.hidden_dim, model_path, basic_cfg.render, train_cfg.randomized_params))  # the args contain shared and not shared
            process.daemon=True  # all processes closed when the main stops
            processes.append(process)

        [p.start() for p in processes]
        while True:  # keep getting the episode reward from the queue
            r = rewards_queue.get()
            rewards.append(r)

            if len(rewards)%20==0 and len(rewards)>0:
                # plot(rewards)
                np.save('log/'+prefix+'td3_rewards', rewards)

        [p.join() for p in processes]  # finished at the same time

        td3_trainer.save_model(model_path)
        
    if basic_cfg.test:
        import time
        model_path = './data/weights/'+ basic_cfg.path +'/{}_td3'.format(str(basic_cfg.model_id))
        print('Load model from: ', model_path)
        td3_trainer.load_model(model_path)
        td3_trainer.to_cuda()
        # print(env.action_space.high, env.action_space.low)

        no_DR = False
        dist_threshold = 0.02
        dist_threshold_max = 0.07
        if no_DR:
            randomized_params=None
        print(randomized_params)
        for eps in range(10):
            if not no_DR:
                param_dict, param_vec = rand_params(env, randomized_params)
                state = env.reset(**param_dict)
                print('Randomized parameters value: ', param_dict)
            else:
                state = env.reset()
            env.render()   
            episode_reward = 0
            import time
            time.sleep(1)
            s_list = []
            for step in range(train_cfg.max_steps):
                action = td3_trainer.policy_net.get_action(state, noise_scale=0.0)

                # pandaopendoorfktactiletest
                # side_action = td3_trainer.side_q.get_action(state)
                # action = np.concatenate([action, [side_action]])


                # offset = np.array([-0.085, 0.])  # offset value when gripper at center of knob but read a non-zero distance
                # norm_dist = np.linalg.norm(np.array(state[21:23])-offset)  # norm of only x- and y-axis
                # gripper_width = state[20]
                # if  norm_dist < dist_threshold:
                #     action[-1] = 0.1
                # elif norm_dist > dist_threshold + 0.01 and gripper_width < dist_threshold_max:
                #     action[-1] = -0.1  # open
                # else:
                #     action[-1] = 0

                next_state, reward, done, _ = env.step(action)
                env.render() 
                # time.sleep(0.1)
                episode_reward += reward
                state=next_state
                # if np.any(state[-30:]>0):  # when there is tactile signal
                #     print(step, state)
                # print(step, action, reward)
                s_list.append(state)
                if done:
                    break

            print('Episode: ', eps, '| Episode Reward: ', episode_reward)
            # np.save('data/s.npy', s_list)
