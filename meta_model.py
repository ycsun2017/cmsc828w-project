import torch
import torch.nn as nn
import argparse
import gym
# import mujoco_py
import numpy as np
from gym.spaces import Box, Discrete
import setup
from algos.memory import Memory, ReplayMemory
from algos.agents.vpg import VPG
from algos.agents.ppo import PPO
from algos.agents.gaussian_vpg import GaussianVPG
from algos.agents.gaussian_model import PolicyHub
from envs.new_cartpole import NewCartPoleEnv
from envs.new_lunar_lander import NewLunarLander
# from stable_baselines.common.env_checker import check_env

import logging
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%m-%d %H:%M:%S")

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default="cpu")
parser.add_argument('--run', type=int, default=-1)
# env settings
parser.add_argument('--env', type=str, default="CartPole-v0")
parser.add_argument('--samples', type=int, default=2000) # need to tune
parser.add_argument('--episodes', type=int, default=10)
parser.add_argument('--steps', type=int, default=300)
parser.add_argument('--goal', type=float, default=0.5) 
parser.add_argument('--seed', default=1, type=int)

# meta settings
parser.add_argument('--meta', dest='meta', action='store_true')
parser.add_argument('--no-meta', dest='meta', action='store_false')
parser.set_defaults(meta=True)
parser.add_argument('--meta-episodes', type=int, default=10)  # need to tune
parser.add_argument('--coeff', type=float, default=0.5)  # need to tune
parser.add_argument('--tau', type=float, default=0.5)  # need to tune

# learner settings
parser.add_argument('--learner', type=str, default="vpg", help="vpg, ppo, sac")
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--update_every', type=int, default=300)
parser.add_argument('--meta_update_every', type=int, default=50)  # need to tune
parser.add_argument('--hiddens', nargs='+', type=int)

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

def make_cart_env(seed):
    # need to tune
    # mass = 0.1 * np.random.randn() + 1.0
    # print("a new env of mass:", mass)
    # env = NewCartPoleEnv(masscart=mass)
    goal = args.goal * np.random.randn() + 0.0
    print("a new env of goal:", goal)
    env = NewCartPoleEnv(goal=goal)
    # check_env(env, warn=True)
    return env

def make_lunar_env(seed):
    # need to tune
    # mass = 0.1 * np.random.randn() + 1.0
    # print("a new env of mass:", mass)
    # env = NewCartPoleEnv(masscart=mass)
    goal = np.random.uniform(-1, 1)
    print("a new env of goal:", goal)
    env = NewLunarLander(goal=goal)
    # check_env(env, warn=True)
    return env

def make_car_env(seed):
    # need to tune
    env = gym.make("MountainCarContinuous-v0")
    return env

if __name__ == '__main__':
    ############## Hyperparameters ##############
    env_name = args.env #"LunarLander-v2"
    samples = args.samples
    max_episodes = args.episodes        # max training episodes
    max_steps = args.steps         # max timesteps in one episode
    meta_episodes = args.meta_episodes
    learner = args.learner
    lr = args.lr
    device = args.device
    update_every = args.update_every
    meta_update_every = args.meta_update_every
    use_meta = args.meta
    coeff = args.coeff
    tau = args.tau
    ############ For All #########################
    gamma = 0.99                # discount factor
    render = False
    save_every = 100
    if args.hiddens:
        hidden_sizes = tuple(args.hiddens) # need to tune
    else:
        hidden_sizes = (32,32)
    activation = nn.Tanh  # need to tune

    use_model = True
    
    torch.cuda.empty_cache()
    ########## file related 
    filename = "usemodel" + env_name + "_" + learner + "_s" + str(samples) + "_n" + str(max_episodes) \
        + "_every" + str(meta_update_every) \
        + "_size" + str(hidden_sizes[0]) + "_c" + str(coeff) + "_tau" + str(tau)
    if not use_meta:
        filename += "_nometa"
    if args.run >=0:
        filename += "_run" + str(args.run)
        
    rew_file = open(args.resdir + filename + ".txt", "w")
    meta_rew_file = open(args.resdir + "meta_" + filename + ".txt", "w")

    # env = gym.make(env_name)
    env = make_cart_env(args.seed)

    if learner == "vpg":
        print("-----initialize meta policy-------")
        meta_policy = GaussianVPG(env.observation_space, env.action_space, meta_update_every,
                hidden_sizes=hidden_sizes, activation=activation, gamma=gamma, device=device, 
                learning_rate=lr, coeff=coeff, tau=tau)
        
    if use_model:
        model_list = []
    for sample in range(samples):
        print("#### Learning environment sample {}".format(sample))
        ########## creating environment
        # env = gym.make(env_name)
        env = make_cart_env(sample)
        # env.seed(sample)
        
        ########## sample a meta learner
        sample_policy = meta_policy.sample_policy()
        # print("-----sample a new policy-------")
        
        ######### single-task learning
        if learner == "vpg":
            actor_policy = VPG(env.observation_space, env.action_space, hidden_sizes=hidden_sizes, 
            activation=activation, gamma=gamma, device=device, learning_rate=lr, with_model=use_model)
            if use_meta:
                actor_policy.set_params(sample_policy)

        memory = Memory()
        if use_model:
            op_memory = ReplayMemory(100000)
        
        all_rewards = []
        start_episode = 0
        timestep = 0
        
        for episode in range(start_episode, max_episodes):
            state = env.reset()
            rewards = []
            for steps in range(max_steps):
                timestep += 1
                
                if render:
                    env.render()
                    
                state_tensor, action_tensor, log_prob_tensor = actor_policy.act(state)
                
                if isinstance(env.action_space, Discrete):
                    action = action_tensor.item()
                else:
                    action = action_tensor.cpu().data.numpy().flatten()
                new_state, reward, done, _ = env.step(action)
                
                rewards.append(reward)
                
                memory.add(state_tensor, action_tensor, log_prob_tensor, reward, done)
                if use_model:
                    op_memory.push(state_tensor, action_tensor, reward, new_state, done)
                # if timestep % update_every == 0: #done or steps == max_steps-1: 
                    
                #     policy_net.update_policy(memory)
                #     memory.clear_memory()
                #     timestep = 0
                    
                state = new_state
                
                if done or steps == max_steps-1:
                    actor_policy.update_policy(memory)
                    memory.clear_memory()
                    # if use_model and op_memory.size() > 256:
                    model_loss = actor_policy.update_model(op_memory)
                        # print(episode, model_loss)
                    all_rewards.append(np.sum(rewards))
                    if episode <= 10:
                        print("sample: {}, episode: {}, total reward: {}".format(
                            sample, episode, np.round(np.sum(rewards), decimals = 3)))
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

        print(sample, "model loss", model_loss)  
        # print("a test")
        # states, actions, rewards, next_states, dones = op_memory.sample(100)
        # actions = np.array([actions]).transpose()
        # pred_delta, pred_rewards, pred_dones = actor_policy.model.predict(states, actions) 
        # print("next state true", next_states)
        # print("next state pred", pred_delta.detach().numpy() + np.array(states))
        # print("rewards true", rewards)
        # print("rewards pred", pred_rewards.detach().numpy().flatten())
        # print("dones true", 1*dones)
        # print("dones pred", pred_dones.detach().numpy().flatten())
        # converted = np.where(pred_dones.detach().numpy().flatten() > 0.5, True, False)
        # print("dones truned", converted)
        # for name, param in actor_policy.model.named_parameters():
        #     print(name, param.data)   
        model_list.append(actor_policy.model)

        if use_model and (sample+1) % meta_update_every == 0:
            print("### Meta Learning")
            meta_policy.meta_update_with_model(model_list, env.reset()) # assume the initial state is the same
            model_list = []
        env.close()

    rew_file.close()
    
            
