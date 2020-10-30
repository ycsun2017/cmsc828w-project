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

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def get_radius_list(radius, rewards):
    r_list = []
    for r in radius:
        r_list.append(rewards[r].mean())
    return r_list

def get_fracs_list(fracs, radius, rewards):
    f_list = []
    for f in fracs:
        f_list.append(rewards[f][radius].mean())
    return f_list

def get_eps_list(rewards):
    return np.mean(rewards, axis=0)

def read_noat(res_dir, env, learner, episodes, runs):
    noat_rewards = []
    for run in range(runs):
        filename = "{}_{}_n{}_run{}.txt".format(env, learner, episodes, run)
        reward = read_rewards(res_dir+filename, episodes)
        noat_rewards.append(reward)
        
    return np.array(noat_rewards)

def read_at(res_dir, env, learner, episodes, at_type, fracs, radius, runs):
    rewards = {}
    for f in fracs:
        rewards[f] = {}
        for r in radius:
            rewards[f][r] = []
            for run in range(runs):
                filename = "{}_{}_n{}_{}_s0.05_m10_r{}_f{}_run{}.txt".format(
                    env, learner, episodes, at_type, np.round(r, decimals = 1), 
                    np.round(f, decimals = 1), run)
                reward = read_rewards(res_dir+filename, episodes)
                rewards[f][r].append(reward)
            rewards[f][r] = np.array(rewards[f][r])
    return rewards

def hybird():
    num_attacks = {}
    for aim in ["action", "reward", "obs"]:
        num_attacks[aim] = 0
    
    with open("results/hybird.txt", "r") as f:
        line = f.readline()
        while line:
            find = re.findall(r'attack  (\w*)', line)
            if find and find[0] != "noat":
                num_attacks[find[0]] += 1
                
            line = f.readline()
    print(num_attacks) # {'action': 295, 'reward': 173, 'obs': 32}


if __name__ == "__main__":
    n = 10
    s = 2000
    xs = list(range(s))
    res = read_rewards("results/CartPole-v0_vpg_s{}_n{}_c0.5.txt".format(s,n), s, n)
    plt.plot(xs, smooth(res, 0.9), label="meta")
    # cs = [0.001, 0.01, 0.1, 0.5, 0.8]
    
    # for c in cs:
    #     r = read_rewards("results/CartPole-v0_vpg_s{}_n{}_c{}.txt".format(s,n,c), s, n)
    #     plt.plot(xs, smooth(r, 0.9),label="meta c="+str(c))
    
#    plt.plot(xs, smooth(at2, 0.9),label="buf")
##    plt.plot(xs, smooth(at3, 0.999),label="action")
##    plt.plot(xs, smooth(at4, 0.999),label="obs")
##    
    plt.legend()
    plt.xlabel("samples (envs)")
    plt.ylabel("reward")
    plt.show()
#    plt.savefig("plots/hybird_vpg_cart.png", format="png")
    
    
    

    
    
    
    
    
    
    
    
    
    
    