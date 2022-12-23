import gym
import torch
import numpy as np
import copy
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from dqn_wrappers import *

def get_state(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0).float()

def select_action(state, policy, eps=0.0):
    sample = random.random()
    if sample > eps:
        with torch.no_grad():
            return policy(state.to(device)).max(1)[1].view(1,1).item()
    else:
        return env.action_space.sample()

def TD0(n_episodes, gamma, alpha0, eta, B, phi0, eps=0.0, cut_factor = 1):
    CIs_Q = np.zeros([n_episodes,2])
    CIs_SE = np.zeros([n_episodes,2])
    Values = np.zeros(n_episodes)
    Theta = np.zeros([B+1,p])
    Thetabar = np.zeros([B+1,p])
    cut = n_episodes // cut_factor
    j=0
    
    for i in tqdm(range(n_episodes)):
        done = False
        state = get_state(env.reset())
        while not done:
            j += 1
            alpha_t = alpha0 * j**(-eta)
            W = np.concatenate([[1], np.random.exponential(size=B)])
            action = select_action(state, policy0, eps=eps)
            obs_, reward, done, info = env.step(action)
            state_ = get_state(obs_)
            phi = model(state.to(device)).detach().to('cpu').numpy().squeeze()
            phi_ = model(state_.to(device)).detach().to('cpu').numpy().squeeze()
            At = np.outer(phi, (phi - gamma*phi_))
            bt = reward*phi
            Theta += alpha_t*np.diag(W) @ (bt[np.newaxis,:] - Theta@At.T)
            if j > cut:
                Thetabar = ((j-cut-1)*Thetabar + Theta)/(j-cut)
            else:
                Thetabar = Theta
            state = state_
        values = Thetabar @ phi0
        value = values[0]
        valuesW = values[1:]
        Q = value + np.quantile(value - valuesW, [0.025, 0.975], axis=0)
        SE = float(np.sqrt(np.cov(value - valuesW)))
        SE = value + 1.96*SE*np.array([-1,1])
        Values[i] = value
        CIs_Q[i,:] = Q
        CIs_SE[i,:] = SE
    
    return {'value': Values, 'Q': CIs_Q, 'SE': CIs_SE}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy0 = torch.load('dqn_pong_model')

    env = gym.make("PongNoFrameskip-v4")
    env = make_env(env)
    nactions = 4

    model = copy.deepcopy(policy0)
    model.head = torch.nn.Identity()

    gamma = 0.99
    p = 512
    n_episodes = 100
    alpha0 = 0.5
    eta = 2/3
    eps=0.0

    state = get_state(env.reset())
    phi0 = model(state.to(device)).detach().to('cpu').numpy().squeeze()

    CIs = TD0(n_episodes, gamma,alpha0,eta,200,phi0)

    values = CIs['value']
    CIs_Q = CIs['Q']
    CIs_SE = CIs['SE']

    # fig 5b
    start=10
    plt.rcParams['figure.figsize'] = (16,9)
    plt.rcParams['figure.facecolor'] = 'white'
    plt.plot(np.arange(n_episodes)[start:], values[start:], color='black', label='value estimate', linewidth=2)
    plt.plot(np.arange(n_episodes)[start:], CIs_Q[start:], color = 'blue', label='Q', linewidth=2)
    plt.plot(np.arange(n_episodes)[start:], CIs_SE[start:], color = 'red', label='SE', linewidth=2)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize='xx-large')
    plt.xlabel('Number of episodes', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig('atari_CI_example.png', bbox_inches='tight')
    plt.close()

    # fig 6a
    CIs_Q_widths = CIs_Q[:,1] - CIs_Q[:,0]
    CIs_SE_widths = CIs_SE[:,1] - CIs_SE[:,0]

    plt.rcParams['figure.figsize'] = (16,9)
    plt.rcParams['figure.facecolor'] = 'white'
    plt.plot(np.arange(n_episodes), CIs_Q_widths, color = 'blue', label='Q', linewidth=2)
    plt.plot(np.arange(n_episodes), CIs_SE_widths, color = 'red', label='SE', linewidth=2)
    plt.legend(fontsize='xx-large')
    plt.xlabel('Number of episodes', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig('atari_CI_widths.png', bbox_inches='tight')
    plt.close()

    # fig 6b
    eps_seq = np.linspace(0,1,5)
    eps_dict = dict()

    for eps in eps_seq:
        eps_dict[eps] = TD0(n_episodes, gamma,alpha0,eta,200,phi0,eps)
        
    SE = np.array([x['SE'] for x in eps_dict.values()])
    Q = np.array([x['Q'] for x in eps_dict.values()])
    values = np.array([x['value'] for x in eps_dict.values()])

    plt.rcParams['figure.figsize'] = (16,9)
    plt.rcParams['figure.facecolor'] = 'white'
    plt.errorbar(eps_seq, values, yerr=Q.T, fmt='.k')
    plt.xlabel('epsilon', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig('atari_CI_bars.png', bbox_inches='tight')
    plt.close()