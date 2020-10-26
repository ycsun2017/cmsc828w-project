import torch
import torch.nn as nn
import argparse
import gym
# import mujoco_py
import numpy as np
from gym.spaces import Box, Discrete
import setup
from algos.memory import Memory
from algos.agents.vpg import VPG
from algos.agents.ppo import PPO
from algos.agents.gaussian_vpg import GaussianVPG
from algos.agents.gaussian_model import PolicyHub
from envs.new_cartpole import NewCartPoleEnv
from stable_baselines.common.env_checker import check_env

import logging
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%m-%d %H:%M:%S")

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default="cpu")
parser.add_argument('--run', type=int, default=-1)
# env settings
parser.add_argument('--env', type=str, default="CartPole-v0")
parser.add_argument('--samples', type=int, default=2000)
parser.add_argument('--episodes', type=int, default=10)
parser.add_argument('--steps', type=int, default=300)

# learner settings
parser.add_argument('--learner', type=str, default="vpg", help="vpg, ppo, sac")
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--update_every', type=int, default=300)
parser.add_argument('--meta_update_every', type=int, default=50)

# file settings
parser.add_argument('--logdir', type=str, default="logs/")
parser.add_argument('--resdir', type=str, default="results/")
parser.add_argument('--moddir', type=str, default="models/")
parser.add_argument('--loadfile', type=str, default="")

args = parser.parse_args()

def get_log(file_name):
    logger = logging.getLogger('train') 
    logger.setLevel(logging.INFO) 

    fh = logging.FileHandler(file_name, mode='a') 
    fh.setLevel(logging.INFO) 
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)  
    return logger

def make_env(seed):
    # mass = 0.1 * np.random.randn() + 1.0
    # print("a new env of mass:", mass)
    # env = NewCartPoleEnv(masscart=mass)
    goal = 0.5 * np.random.randn() + 0.0
    print("a new env of goal:", goal)
    env = NewCartPoleEnv(goal=goal)
    check_env(env, warn=True)
    return env

if __name__ == '__main__':
    ############## Hyperparameters ##############
    env_name = args.env #"LunarLander-v2"
    samples = args.samples
    max_episodes = args.episodes        # max training episodes
    max_steps = args.steps         # max timesteps in one episode
    learner = args.learner
    
    lr = args.lr
    device = args.device
    ############ For All #########################
    gamma = 0.99                # discount factor
    random_seed = 0 
    render = False
    update_every = args.update_every
    save_every = 100
    meta_update_every = args.meta_update_every

    torch.cuda.empty_cache()
    ########## file related 
    filename = env_name + "_" + learner + "_s" + str(samples) + "_n" + str(max_episodes)
    if args.run >=0:
        filename += "_run" + str(args.run)
        
    rew_file = open(args.resdir + filename + ".txt", "w")

    # env = gym.make(env_name)
    env = make_env(0)

    if learner == "vpg":
        print("-----initialize meta policy-------")
        meta_policy = GaussianVPG(env.observation_space, env.action_space, meta_update_every,
                hidden_sizes=(4,4), activation=nn.Tanh, gamma=gamma, device=device, learning_rate=lr)

    meta_memory = Memory()
    for sample in range(samples):
        print("#### Learning environment sample {}".format(sample))
        ########## creating environment
        env = make_env(sample)
        # env.seed(sample)
        
        ########## sample a learner
        policy_net = meta_policy.sample_policy()
        print("-----sample a new policy-------")
        print("weight of layer 0", policy_net.action_layer[0].weight) 

        start_episode = 0
        # load learner from checkpoint
        if args.loadfile != "":
            checkpoint = torch.load(args.moddir + args.loadfile)
            print("load from ", args.moddir + args.loadfile)
            policy_net.set_state_dict(checkpoint['model_state_dict'], checkpoint['optimizer_state_dict'])
            start_episode = checkpoint['episode']
        
        memory = Memory()
        
        all_rewards = []
        timestep = 0
        
        ######### training
        for episode in range(start_episode, max_episodes):
            state = env.reset()
            rewards = []
            for steps in range(max_steps):
                timestep += 1
                
                if render:
                    env.render()
                    
                state_tensor, action_tensor, log_prob_tensor = policy_net.act(state, device)
                
                if isinstance(env.action_space, Discrete):
                    action = action_tensor.item()
                else:
                    action = action_tensor.cpu().data.numpy().flatten()
                new_state, reward, done, _ = env.step(action)
                # print("state", new_state, "reward", reward)
                rewards.append(reward)
                
                # memory.add(state_tensor, action_tensor, log_prob_tensor, reward, done)
                # if episode == start_episode:
                meta_memory.add(state_tensor, action_tensor, log_prob_tensor, reward, done)
                
                # if timestep % update_every == 0: #done or steps == max_steps-1: 
                    
                #     policy_net.update_policy(memory)
                #     memory.clear_memory()
                #     timestep = 0
                    
                state = new_state
                
                if done or steps == max_steps-1:
                    all_rewards.append(np.sum(rewards))
    #                 logger.info("episode: {}, total reward: {}\n".format(episode, np.round(np.sum(rewards), decimals = 3)))
                    rew_file.write("sample: {}, episode: {}, total reward: {}\n".format(
                        sample, episode, np.round(np.sum(rewards), decimals = 3)))
                    break
                # if (episode+1) % save_every == 0:
                #     path = args.moddir + filename
                #     torch.save({
                #     'episode': episode,
                #     'model_state_dict': policy_net.get_state_dict()[0],
                #     'optimizer_state_dict': policy_net.get_state_dict()[1]
                #     }, path)
                
        if (sample+1) % meta_update_every == 0:
            meta_policy.meta_update(meta_memory)
            meta_memory.clear_memory()

        env.close()

    rew_file.close()
    
            
