import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
from tqdm.notebook import tqdm
from functools import partial
import datetime
from scipy.linalg import eig
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import ray
import psutil

def flatten_mdp(policy, model):
    nstates = policy.shape[0] 
    P_pi = np.zeros([nstates, nstates])  # transition probability matrix (s) to (s')
    R_pi = np.zeros([nstates])             # exp. reward from state (s) to any next state
    for s in range(nstates):
        for a in range(nactions):
            for p_, s_, r_, _ in model[s][a]:
                P_pi[s, s_] += policy[s,a] * p_   # transition probability (s) -> (s')
                Rsa = p_ * r_                     # exp. reward from (s,a) to any next state
                R_pi[s] += policy[s,a] * Rsa      # exp. reward from (s) to any next state
    assert np.allclose(P_pi.sum(axis=1),1)  # rows should sum to 1
    return P_pi, R_pi
    
def gen_data(env, policy, n_games, Phi):
    Data = []
    nactions = env.env.nA
    for i in range(n_games):
        data = []
        done=False
        state=env.reset()
        while not done:
            action = np.random.choice(np.arange(nactions), p=policy[state,:])
            state_, reward, done, info = env.step(action)
            data.append((state,reward,state_))
            state = state_
        Data.append(data)
    return Data

def online_bootstrap(Data, Phi, gamma, alpha0, eta, B, cut_factor=1, freq=10):
    CIs = []
    Theta = Thetabar = np.zeros([B+1,Phi.shape[1]])
    n_games = len(Data)
    cut = n_games//cut_factor
    j=0
    
    for i in range(n_games):
        data = Data[i]
        for k in range(len(data)):
            j+=1
            alpha_t = alpha0 * j**(-eta)
            W = np.concatenate([[1], np.random.exponential(size=B)])
            state,reward,state_ = data[k]
            phi = Phi[state,:]
            phi_ = Phi[state_,:]
            At = np.outer(phi, (phi-gamma*phi_))
            bt = reward*phi
            Theta += alpha_t*np.diag(W) @ (bt[np.newaxis,:] - Theta@At.T)
            if j > cut:
                Thetabar = ((j-cut-1)*Thetabar + Theta)/(j-cut)
            else:
                Thetabar = Theta
        if i % freq == 0:
            thetabar = Thetabar[0,:]
            ThetabarW = Thetabar[1:,:]    
            SE = np.sqrt(np.diag(Phi @ np.cov(thetabar[np.newaxis,:] - ThetabarW, rowvar=False) @ Phi.T))
            CI = (Phi @ thetabar)[:,np.newaxis] + 1.96*np.outer(SE, [-1,1])
            CI = np.insert(CI,1,Phi@thetabar,axis=1)
            CIs.append(CI)
    return CIs

def offline_bootstrap(Data, Phi, gamma, alpha0, eta, B, cut_factor=1, freq=10):
    CIs = []
    Theta = Thetabar = np.zeros([B+1,Phi.shape[1]])
    n_games = len(Data)
    samples = [np.arange(n_games)] + [np.random.choice(n,size=n,replace=True) for _ in range(B)]
    cut = n_games//cut_factor
    js=np.zeros(B+1).astype(int)
    
    for i in range(n_games):
        for b in range(B+1):
            data = Data[samples[b][i]]
            for k in range(len(data)):
                js[b]+=1
                alpha_t = alpha0 * js[b]**(-eta)
                state,reward,state_ = data[k]
                phi = Phi[state,:]
                phi_ = Phi[state_,:]
                At = np.outer(phi, (phi-gamma*phi_))
                bt = reward*phi
                Theta[b,:] += alpha_t*(bt - At@Theta[b,:])
                if js[b] > cut:
                    Thetabar[b,:] = ((js[b]-cut-1)*Thetabar[b,:] + Theta[b,:])/(js[b]-cut)
                else:
                    Thetabar[b,:] = Theta[b,:]
        
        if i % freq == 0:
            thetabar = Thetabar[0,:]
            ThetabarW = Thetabar[1:,:]    
            SE = np.sqrt(np.diag(Phi @ np.cov(thetabar[np.newaxis,:] - ThetabarW, rowvar=False) @ Phi.T))
            CI = (Phi @ thetabar)[:,np.newaxis] + 1.96*np.outer(SE, [-1,1])
            CI = np.insert(CI,1,Phi@thetabar,axis=1)
            CIs.append(CI)
    return CIs

@ray.remote
def run_bootstrap(seed, env, policy, n_games, Phi, gamma, alpha0, eta, B, cut_factor=1, freq=10):
    np.random.seed(seed)
    if seed % 50 == 0:
        now = dt.datetime.now()
        print (seed, now.strftime("%Y-%m-%d %H:%M:%S"))
    Data = gen_data(env, policy, n_games, Phi)
    CIs_online = online_bootstrap(Data, Phi, gamma, alpha0, eta, B, cut_factor, freq)
    CIs_offline = online_bootstrap(Data, Phi, gamma, alpha0, eta, B, cut_factor, freq)
    return {'online':CIs_online, 'offline':CIs_offline}

if __name__ == "__main__":
    num_cpus = psutil.cpu_count(logical=False)
    ray.init(num_cpus=num_cpus, ignore_reinit_error=True)

    np.random.seed(2021)
    random_map = generate_random_map(size=8,p=0.8)
    env = gym.make('FrozenLake-v0', desc=random_map)

    nstates = env.env.nS
    nactions = env.env.nA
    P_MDP = env.env.P

    # generate feature matrix (row n corresponds to state n)
    gamma = 0.99
    p = 4
    Phi = np.random.random(size=(nstates, p))
    
    # create policy matrix from Q table
    Q_table = np.loadtxt('Q_fz.csv', delimiter=',')
    policy_Q = np.array([np.eye(nactions)[np.argmax(row),:] if row.sum() > 0 else np.ones(nactions)/nactions for row in Q_table])
    policy_random = np.ones([nstates, nactions]) / nactions  # 0.25 probability for each action
    
    epsilon = 0.2
    policy = (1-epsilon)*policy_Q + epsilon*policy_random

    P_pi, R_pi = flatten_mdp(policy, model=P_MDP)
    Abar = np.eye(nstates) - gamma*P_pi
    bbar = R_pi

    # convert to infinite horizon TPM
    for i in range(nstates):
        if P_pi[i,i] == 1.0:
            P_pi[i,i] = 0
            P_pi[i,0] = 1.0
            
    eigen = eig(P_pi,left=True)
    tmp = np.real(eigen[1][:,np.argmin(np.abs(eigen[0] - 1))])
    stat_dist = tmp / np.sum(tmp)
    D_pi = np.diag(stat_dist)

    Abar = Phi.T @ D_pi @ (np.eye(nstates) - gamma*P_pi) @ Phi
    bbar = Phi.T @ D_pi @ R_pi
    theta0 = np.linalg.pinv(Abar) @ bbar # analytical solution
    value0 = np.dot(Phi, theta0) # true value functions

    # params
    niter = 200
    n_games = int(2e3)
    alpha0 = 5
    eta = 2/3
    B = 200
    args = {'env':env, 'policy':policy, 'n_games':n_games, 'Phi':Phi, 'gamma':gamma, 'alpha0':alpha0, 'eta':eta, 'B':B, 
            'cut_factor':1, 'freq':10,
           }

    res = ray.get([run_bootstrap.remote(seed, **args) for seed in range(niter)])

    online = [r['online'] for r in res]
    offline = [r['offline'] for r in res]

    #coverage
    dim = 0
    cov_online = np.array([np.array([(value0[dim] > CIs[dim,0]) & (value0[dim] < CIs[dim,2]) for CIs in r]) for r in online]).mean(axis=0)
    cov_offline = np.array([np.array([(value0[dim] > CIs[dim,0]) & (value0[dim] < CIs[dim,2]) for CIs in r]) for r in offline]).mean(axis=0)
    x = np.arange(len(cov_online))*10

    plt.rcParams['figure.figsize'] = (16,9)
    plt.rcParams['figure.facecolor'] = 'white'
    plt.plot(x,cov_online, label='online', alpha=0.8, color='blue', linewidth=2)
    plt.plot(x,cov_offline, label='offline', alpha=0.8, color='red', linewidth=2)
    plt.axhline(0.95, linestyle='--', color='black', linewidth=2)
    plt.ylim((0,1))
    plt.xlabel('Number of episodes', fontsize=20)
    plt.ylabel('CI Coverage ({} runs)'.format(niter), fontsize=20)
    plt.legend(fontsize='xx-large')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig('frozenlake_value0_coverage.png', bbox_inches='tight')
    plt.close()

    # CI example
    start=50

    run = np.argmin([(online[run][-1][dim,1] - value0[dim])**2 for run in range(niter)])

    CIs_online = np.array([CIs[dim,[0,2]] for CIs in online[run]])
    CIs_offline = np.array([CIs[dim,[0,2]] for CIs in offline[run]])
    values_online = np.array([CIs[dim,1] for CIs in online[run]])
    values_offline = np.array([CIs[dim,1] for CIs in offline[run]])

    plt.rcParams['figure.figsize'] = (16,9)
    plt.rcParams['figure.facecolor'] = 'white'
    plt.plot(x[start:],CIs_online[start:], color='blue', label='online', linewidth=2)
    plt.plot(x[start:],CIs_offline[start:], color='red', label='offline', linewidth=2)
    plt.plot(x[start:],values_online[start:],color='black',label='value estimate', linewidth=2)
    plt.axhline(value0[dim],color='black', linestyle='--', label='true value function', linewidth=2)
    plt.xlabel('Number of episodes', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    legend = [Line2D([0],[0],color='blue',label='online', linewidth=2),
              Line2D([0],[0],color='red',label='offline', linewidth=2),
              Line2D([0],[0],color='black',label='value estimate', linewidth=2),
              Line2D([0],[0],color='black',linestyle='--',label='true value function', linewidth=2),
             ]
    plt.legend(handles=legend, loc='upper right', fontsize='xx-large')
    plt.savefig('frozenlake_value0_CI_example.png', bbox_inches='tight')
    plt.close()