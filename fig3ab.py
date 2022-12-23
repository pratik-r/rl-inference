import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
from scipy.linalg import eig
from tqdm import tqdm
import matplotlib.pyplot as plt

def rwfn(n_states):
    states = np.random.randn(n_states)
    tpm = np.diag(np.random.random(n_states)*0.5)
    indices = np.indices(np.shape(tpm))
    rows = indices[0]
    columns = indices[1]
    tpm[rows == (columns-1)] = (1-np.diag(tpm)[:(n_states-1)])/2
    tpm[rows == (columns+1)] = (1-np.diag(tpm)[1:])/2
    tpm[0,1] = 1-tpm[0,0]
    tpm[-1,-2] = 1-tpm[-1,-1]
    eigen = eig(tpm,left=True)
    tmp = eigen[1][:,np.argmin(np.abs(eigen[0] - 1))]
    stat_dist = tmp / np.sum(tmp)
    return dict(states=states, tpm=tpm, stat_dist=stat_dist)

def sim_path(n, states, tpm, stat_dist):
    path_idx = np.zeros(n).astype(int)
    n_states = len(states)
    path_idx[0] = np.random.choice(np.arange(n_states), size=1, p=stat_dist)
    for i in range(1,n):
        path_idx[i] = np.random.choice(np.arange(n_states), size=1, p=tpm[path_idx[i-1],:])
    path = states[path_idx]
    return dict(idx=path_idx,path=path)

def gen_linear(p,q,n,theta0,states,tpm,stat_dist):
    X = np.random.randn(n,p)
    eps = sim_path(n,states,tpm,stat_dist)['path']
    y = X@theta0 + eps
    return dict(X=X, y=y)

def online_bootstrap(dat, B, alpha0, eta, cut_factor=5, freq = 10):
    n,p = dat['X'].shape
    CIs = {'Q': [], 'SE': []}
    cut = n//cut_factor
    Theta = Thetabar = np.zeros([B+1,p])
    for i in tqdm(range(n)):
        alpha_t = alpha0 * (i+1)**(-eta)
        W = np.concatenate([[1], np.random.exponential(size=B)])
        Theta += 2*alpha_t*np.diag(W) @ np.outer((np.repeat(dat['y'][i],B+1) - Theta @ dat['X'][i,:]), dat['X'][i,:])
        if i > cut:
            Thetabar = ((i-cut-1)*Thetabar + Theta)/(i-cut)
        else:
            Thetabar=Theta
        if i % freq == 0:
            thetabar = Thetabar[0,:]
            ThetabarW = Thetabar[1:,:]    
            Q = thetabar[:,np.newaxis] + np.quantile(thetabar[np.newaxis,:] - ThetabarW, [0.025,0.975], axis=0).T
            SE = np.sqrt(np.diag(np.cov(thetabar[np.newaxis,:] - ThetabarW, rowvar=False)))
            SE = np.outer(thetabar, np.ones(2)) + 1.96*np.outer(SE, [-1,1])
            Q = np.insert(Q,1,thetabar,axis=1)
            SE = np.insert(SE,1,thetabar,axis=1)
            CIs['Q'].append(Q)
            CIs['SE'].append(SE)
    return CIs
    
if __name__ == "__main__":
    np.random.seed(2021)
    n_states = 50
    rw = rwfn(n_states)
    states = rw['states']; tpm = rw['tpm']; stat_dist=rw['stat_dist']
    niter=400
    n=int(4e3)
    mu = 0.2
    p=10
    q=6
    B=200
    alpha0=0.5
    eta=0.75
    block_length=1
    freq = 10
    theta0 = np.concatenate((np.repeat(mu,np.round(q/2)), np.repeat(-mu,q-np.round(q/2)), np.repeat(0,p-q)))
    args = {'n': n,'p': p,'q': q,'B': B,'alpha0': alpha0,'eta': eta,'theta0': theta0,
            'states':states,'tpm':tpm,'stat_dist':stat_dist, 'block_length': block_length,
    }
    
    dat = gen_linear(p,q,n,theta0,states,tpm,stat_dist)
    CIs = online_bootstrap(dat, B, alpha0, eta, cut_factor=5, freq = freq)
    
    dim = 0
    start = 100
    
    alpha0s = [0.1,0.5,1.0]
    etas = [0.6,0.75,0.9]
    cols = ['blue', 'green', 'red']
    
    # step size sensitivity
    plt.rcParams['figure.figsize'] = (16,9)
    plt.rcParams['figure.facecolor'] = 'white'

    for i in range(len(alpha0s)):
        CIs = online_bootstrap(dat, B, alpha0s[i], eta, cut_factor=5, freq = 10)
        CI_widths = np.array([CI[dim,2]-CI[dim,0] for CI in CIs['Q']])
        plt.plot((freq*np.arange(len(CI_widths)))[start:], CI_widths[start:], label='alpha0 = {0:.1f}'.format(alpha0s[i]), color=cols[i], linewidth=3)
        
    plt.xlabel('Number of episodes', fontsize=20)
    plt.ylabel('CI width', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize='x-large')
    plt.savefig('markov_rw_stepsize.png', bbox_inches='tight')
    plt.close()
    
    # learning rate sensitivity
    plt.rcParams['figure.figsize'] = (16,9)
    plt.rcParams['figure.facecolor'] = 'white'

    for i in range(len(alpha0s)):
        CIs = online_bootstrap(dat, B, alpha0, etas[i], cut_factor=5, freq = 10)
        CI_widths = np.array([CI[dim,2]-CI[dim,0] for CI in CIs['Q']])
        plt.plot((freq*np.arange(len(CI_widths)))[start:], CI_widths[start:], label='eta = {0:.2f}'.format(etas[i]), color=cols[i], linewidth=3)
        
    plt.xlabel('Number of episodes', fontsize=20)
    plt.ylabel('CI width', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize='x-large')
    plt.savefig('markov_rw_learning_rate.png', bbox_inches='tight')
    plt.close()