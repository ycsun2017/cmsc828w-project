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

def read_rewards_multi(samples, episodes, coeff, runs, nometa=False):
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
    runs = 10
    xs = list(range(s))
    for tau in [0.1,0.5,0.8,0.9]:
        res = read_rewards("results/CartPole-v0_vpg_s{}_n{}_goal0.5_c0.5_tau{}.txt".format(s,n,tau), s, n)
        plt.plot(xs, smooth(res, 0.999), label="tau"+str(tau))
    # res = read_rewards("results/CartPole-v0_vpg_s{}_n{}_goal0.5_c0.5_tau0.5.txt".format(s,n), s, n)
    # plt.plot(xs, smooth(res, 0.999), label="tau0.5")
    # res = read_rewards("results/CartPole-v0_vpg_s{}_n{}_goal0.5_c0.5_tau0.8.txt".format(s,n), s, n)
    # plt.plot(xs, smooth(res, 0.999), label="tau0.8")

    # nometa = read_rewards_multi(s,n,0.5,runs,nometa=True)
    # plt.plot(xs, smooth(nometa, 0.9), label="meta")
    # cs = [0.5]
    # for c in cs:
    #     mean_rewards = read_rewards_multi(s,n,c,runs)
    #     # r = read_rewards("results/CartPole-v0_vpg_s{}_n{}_goal0.3_c{}.txt".format(s,n,c), s, n)
    #     plt.plot(xs, smooth(mean_rewards, 0.9),label="meta c="+str(c))
    
#    plt.plot(xs, smooth(at2, 0.9),label="buf")
##    plt.plot(xs, smooth(at3, 0.999),label="action")
##    plt.plot(xs, smooth(at4, 0.999),label="obs")
##    
    plt.legend()
    plt.xlabel("samples (envs)")
    plt.ylabel("reward")
    plt.show()
#    plt.savefig("plots/hybird_vpg_cart.png", format="png")
    
    
    

    
    
    
    
    
    
    
    
    
    
    