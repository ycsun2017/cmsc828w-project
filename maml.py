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
# from envs.swimmer_rand_vel import SwimmerEnvRandVel
# from stable_baselines.common.env_checker import check_env

import logging
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%m-%d %H:%M:%S")

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default="cuda")
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
parser.add_argument('--resdir', type=str, default="results_peihong/")
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


if __name__ == '__main__':
    ############## Hyperparameters ##############
    env_name = args.env  # "LunarLander-v2"
    samples = args.samples
    max_episodes = args.episodes  # max training episodes
    max_steps = args.steps  # max timesteps in one episode
    meta_episodes = args.meta_episodes
    learner = args.learner
    lr = args.lr
    device = args.device
    update_every = args.update_every
    meta_update_every = args.meta_update_every
    use_meta = args.meta
    coeff = args.coeff
    ############ For All #########################
    gamma = 0.99  # discount factor
    render = False
    save_every = 100
    hidden_sizes = (16, 16)  # need to tune
    activation = nn.Tanh  # need to tune

    torch.cuda.empty_cache()
    ########## file related
    filename = env_name + "_" + learner + "_s" + str(samples) + "_n" + str(max_episodes) + "_c" + str(coeff)
    if args.run >= 0:
        filename += "_run" + str(args.run)

    rew_file = open(args.resdir + filename + ".txt", "w")
    meta_rew_file = open(args.resdir + "meta_" + filename + ".txt", "w")

    # env = gym.make(env_name)
    env = make_cart_env(args.seed)

    if learner == "vpg":
        actor_policy = VPG(env.observation_space, env.action_space, hidden_sizes=hidden_sizes,
                           activation=activation, gamma=gamma, device=device, learning_rate=lr, with_meta=True)

    for sample in range(samples):
        print("sample " + str(sample))
        env = make_cart_env(sample)

        meta_memory = Memory()
        memory = Memory()

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

                state = new_state

                if done or steps == max_steps - 1:
                    actor_policy.update_policy_m(memory)
                    memory.clear_memory()
                    rew_file.write("sample: {}, episode: {}, total reward: {}\n".format(
                        sample, episode, np.round(np.sum(rewards), decimals=3)))
                    break

        state = env.reset()
        rewards = []
        for steps in range(max_steps):
            if render:
                env.render()

            state_tensor, action_tensor, log_prob_tensor = actor_policy.act_policy_m(state)

            if isinstance(env.action_space, Discrete):
                action = action_tensor.item()
            else:
                action = action_tensor.cpu().data.numpy().flatten()
            new_state, reward, done, _ = env.step(action)

            rewards.append(reward)
            meta_memory.add(state_tensor, action_tensor, log_prob_tensor, reward, done)
            state = new_state

            if done or steps == max_steps - 1:
                meta_rew_file.write("sample: {}, episode: {}, total reward: {}\n".format(
                            sample, episode, np.round(np.sum(rewards), decimals=3)))
                break

        if (sample+1) % meta_update_every == 0:
            actor_policy.update_policy(meta_memory)
            meta_memory.clear_memory()

        env.close()

    rew_file.close()
    meta_rew_file.close()


