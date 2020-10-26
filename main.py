import torch
import argparse
import gym
import mujoco_py
import numpy as np
from gym.spaces import Box, Discrete
import setup
from poison_rl.memory import Memory
from poison_rl.agents.vpg import VPG
from poison_rl.agents.ppo import PPO
from poison_rl.attackers.wb_attacker import WbAttacker
from poison_rl.attackers.rand_attacker import RandAttacker

import logging
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%m-%d %H:%M:%S")

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default="cpu")
parser.add_argument('--run', type=int, default=-1)
# env settings
parser.add_argument('--env', type=str, default="CartPole-v0")
parser.add_argument('--episodes', type=int, default=1000)
parser.add_argument('--steps', type=int, default=300)

# learner settings
parser.add_argument('--learner', type=str, default="vpg", help="vpg, ppo, sac")
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--update_every', type=int, default=300)

# attack settings
parser.add_argument('--norm', type=str, default="l2")
parser.add_argument('--stepsize', type=float, default=0.05)
parser.add_argument('--maxiter', type=int, default=10)
parser.add_argument('--radius', type=float, default=0.5)
parser.add_argument('--frac', type=float, default=1.0)
parser.add_argument('--dist-thres', type=float, default=0.1)
parser.add_argument('--type', type=str, default="wb", help="rand, wb, semirand")

parser.add_argument('--attack', dest='attack', action='store_true')
parser.add_argument('--no-attack', dest='attack', action='store_false')
parser.set_defaults(attack=True)

parser.add_argument('--compute', dest='compute', action='store_true')
parser.add_argument('--no-compute', dest='compute', action='store_false')
parser.set_defaults(compute=False)

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


if __name__ == '__main__':
    ############## Hyperparameters ##############
    env_name = args.env #"LunarLander-v2"
    
    max_episodes = args.episodes        # max training episodes
    max_steps = args.steps         # max timesteps in one episode
    attack = args.attack
    compute = args.compute
    attack_type = args.type
    learner = args.learner
    
    stepsize = args.stepsize
    maxiter = args.maxiter
    radius = args.radius
    frac = args.frac
    lr = args.lr
    device = args.device
    ############ For All #########################
    gamma = 0.99                # discount factor
    random_seed = 0 
    render = False
    update_every = args.update_every
    save_every = 100
    
    ########## creating environment
    env = gym.make(env_name)
    
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
    
    
    ########## file related 
    filename = env_name + "_" + learner + "_n" + str(max_episodes)
    if attack:
        filename += "_" + attack_type
        filename += "_s" + str(stepsize) + "_m" + str(maxiter) + "_r" + str(radius) + "_f" + str(frac)
    if args.run >=0:
        filename += "_run" + str(args.run)
        
        
#     logger = get_log(args.logdir + filename + "_" +current_time)
#     logger.info(args)
    
    rew_file = open(args.resdir + filename + ".txt", "w")
    if compute:
        radius_file = open(args.resdir + filename + "_radius" + "_s" + str(stepsize) + "_m" + str(maxiter) + "_r" + str(radius) + "_th" + str(args.dist_thres) + ".txt", "w")
    
    
    ########## create learner
    if learner == "vpg":
        policy_net = VPG(env.observation_space, env.action_space, gamma=gamma, device=device, learning_rate=lr)
    elif learner == "ppo":
        policy_net = PPO(env.observation_space, env.action_space, gamma=gamma, device=device, learning_rate=lr)
    
    
    ########## create attacker
    if attack_type == "wb":
        attack_net = WbAttacker(env.observation_space, env.action_space, policy_net, maxat=int(frac*max_episodes), maxeps=max_episodes,
                                gamma=gamma, learning_rate=lr, maxiter=maxiter, radius=radius, stepsize=stepsize, dist_thres=args.dist_thres)
    elif attack_type == "rand":
        attack_net = RandAttacker(radius=radius, frac=frac, maxat=int(frac*max_episodes))
    elif attack_type == "semirand":
        attack_net = WbAttacker(env.observation_space, env.action_space, policy_net, maxat=int(frac*max_episodes), maxeps=max_episodes,
                                gamma=gamma, learning_rate=lr, maxiter=maxiter, radius=radius, stepsize=stepsize, rand_select=True)
            
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
                
            state_tensor, action_tensor, log_prob_tensor = policy_net.act(state)
            
            if isinstance(env.action_space, Discrete):
                action = action_tensor.item()
            else:
                action = action_tensor.cpu().data.numpy().flatten()
            new_state, reward, done, _ = env.step(action)
            
            rewards.append(reward)
            
            memory.add(state_tensor, action_tensor, log_prob_tensor, reward, done)
            
            if timestep % update_every == 0: #done or steps == max_steps-1: 
                if attack:
                    if attack_type == "bb": # and attack_net.buffer.size() > 128:
                        attack_net.learning()
                    attack_r = attack_net.attack_r_general(memory)
#                     logger.info(memory.rewards)
                    memory.rewards = attack_r.copy()
#                     logger.info(memory.rewards)
                if compute:
                    disc = attack_net.compute_disc(memory)
                    print("policy discrepancy:", disc)
                    radius_file.write("disc: {}\n".format(disc))
                policy_net.update_policy(memory)
                memory.clear_memory()
                timestep = 0
                
            state = new_state
            
            if done or steps == max_steps-1:
                all_rewards.append(np.sum(rewards))
#                 logger.info("episode: {}, total reward: {}\n".format(episode, np.round(np.sum(rewards), decimals = 3)))
                rew_file.write("episode: {}, total reward: {}\n".format(episode, np.round(np.sum(rewards), decimals = 3)))
                break
            if (episode+1) % save_every == 0:
                path = args.moddir + filename
                torch.save({
                   'episode': episode,
                   'model_state_dict': policy_net.get_state_dict()[0],
                   'optimizer_state_dict': policy_net.get_state_dict()[1]
                   }, path)
    if attack:
        logger.info("total attacks: {}\n".format(attack_net.attack_num))
        print("total attacks: {}\n".format(attack_net.attack_num))
        
    rew_file.close()
    if compute:
        radius_file.close()
    env.close()
            
