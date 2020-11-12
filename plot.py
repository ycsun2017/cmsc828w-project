import matplotlib.pyplot as plt
import tikzplotlib
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
    runs = 10
    xs = list(range(s))

    SMALL_SIZE = 20
    MEDIUM_SIZE = 25
    BIGGER_SIZE = 25

    # plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    # plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.rcParams["figure.figsize"] = (10,8)

# for no_meta
    # for tau in [0.8]:
    #     for every in [50]:
    #         res = read_rewards_multi("results/n50/Lunar_vpg_s{}_n{}_every{}_size32_c0.5_tau{}".format(s,n,every,tau), s, n, runs)
    #         plt.plot(xs, smooth(res, 0.99), label="meta")
    #
    #
    # for tau in [0.8]:
    #     for every in [50]:
    #         res = read_rewards_multi("results/Lunar_vpg_s{}_n{}_every{}_size32_c0.5_tau{}_nometa".format(s,n,every,tau), s, n, runs)
    #         plt.plot(xs, smooth(res, 0.99), label="no_meta")

#meta
    # for tau in [0.8]:
    #     for every in [50]:
    #         res = read_rewards_multi("results/n50/Lunar_vpg_s{}_n{}_every{}_size32_c0.5_tau{}".format(s,n,every,tau), s, n, runs)
    #         plt.plot(xs, smooth(res, 0.99), label="every" + str(every))


## Swimmer
    plt.style.use("ggplot")
    dirname = "results_swimmer/"
    s = 1000
    xs = list(range(s))
    res = read_rewards_multi(dirname+"Swimmer_vpg_s{}_n{}_every50_size32_c0.5_tau0.5_nometa".format(s,n), s, n, runs)
    plt.plot(xs, smooth(res, 0.99), label="single-task")

    for tau in [0.5]:
        for every in [25, 50, 75]:
            res = read_rewards_multi(dirname+"Swimmer_vpg_s{}_n{}_every{}_size32_c0.5_tau{}".format(s,n,every,tau), s, n, runs)
            plt.plot(xs, smooth(res, 0.99), label="meta_N="+str(every))




    plt.legend()
    plt.xlabel("Tasks (environments)")
    plt.ylabel("Mean reward")
    # plt.show()
    plt.savefig("plots/swimmer.eps", format="eps")
    tikzplotlib.save("plots/swimmer.tex")
    

    
    
    
    
    
    
    
    
    
    
    
