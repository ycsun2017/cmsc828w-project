import matplotlib.pyplot as plt
import numpy as np
import re
import math


def read_rewards(filename, samples, episodes):
    rewards = []
    with open(filename, "r") as f:
        for i in range(samples):
            rew_sum = 0
            for j in range(episodes):
                line = f.readline()
                rew = float(line.split()[-1])
                rew_sum += rew
            rewards.append(rew_sum / episodes)
    return rewards

def read_rewards_multi(filename, samples, episodes, runs):
    rewards = []
    for run in range(runs):
        reward = read_rewards(filename+"_run{}.txt".format(run), samples,episodes)
        rewards.append(reward)
    rewards = np.array(rewards)
    # print("rewards", rewards)
    return np.mean(rewards, axis=0)

def read_rewards_multi_old(samples, episodes, coeff, runs, nometa=False):
    rewards = []
    for run in range(runs):
        if nometa:
            reward = read_rewards("results/CartPole-v0_vpg_s{}_n{}_goal0.5_c{}_nometa_run{}.txt".format(
            samples,episodes,coeff, run), samples,episodes)
        else:
            reward = read_rewards("results/CartPole-v0_vpg_s{}_n{}_goal0.5_c{}_run{}.txt".format(
                samples,episodes,coeff, run), samples,episodes)
        rewards.append(reward)
    rewards = np.array(rewards)
    # print("rewards", rewards)
    return np.mean(rewards, axis=0)

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


if __name__ == "__main__":
    n = 10
    s = 2000
    runs = 2
    xs = list(range(s))
    # for tau in [0.5]:
        # for every in [25,50]:
        #     res = read_rewards_multi("results/CartPole-v0_vpg_s{}_n{}_every{}_goal0.5_c0.5_tau{}".format(s,n,every,tau), s, n, runs)
        #     plt.plot(xs, smooth(res, 0.99), label="every"+str(every))
        # for every in [25,75]:
        #     res = read_rewards_multi("results/Swimmer_vpg_s{}_n{}_every{}_size32_c0.5_tau{}".format(s,n,every,tau), s, n, runs)
        #     plt.plot(xs, smooth(res, 0.9), label="every"+str(every))

    # for tau in [0.5, 0.8]:
    #     for every in [50]:
    #         res = read_rewards("results/Swimmer_vpg_s{}_n{}_every{}_goal0.5_c0.5_tau{}.txt".format(s,n,every,tau), s, n)
    #         plt.plot(xs, smooth(res, 0.99), label="tau"+str(tau))

    
    # for tau in [0.8]:
    #     for every in [25,75]:
    #         res = read_rewards("results/Swimmer_vpg_s{}_n{}_every{}_size32_c0.5_tau{}.txt".format(s,n,every,tau), s, n)
    #         plt.plot(xs, smooth(res, 0.99), label="every"+str(every))
    res = read_rewards("results_peihong/maml_CartPole-v0_vpg_s2000_n10_every75_size32.txt", s, n)
    plt.plot(xs, smooth(res, 0.99), label="nometa")


    # res = read_rewards("results_peihong/CartPole-v0_vpg_s2000_n10_c0.5.txt", s, n)
    # plt.plot(xs, smooth(res, 0.99))

##    
    plt.legend()
    plt.xlabel("samples (envs)")
    plt.ylabel("reward")
    plt.show()
#    plt.savefig("plots/hybird_vpg_cart.png", format="png")
    
    
    

    
    
    
    
    
    
    
    
    
    
    