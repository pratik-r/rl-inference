import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
from scipy.linalg import eig
from tqdm.notebook import tqdm
from functools import partial
import ray
import psutil
import datetime as dt
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
    for i in range(n):
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

# moving block bootstrap
def mb_bootstrap(dat, B, alpha0, eta, cut_factor=5, freq=10, block_length=10):
    n,p = dat['X'].shape
    CIs = {'Q': [], 'SE': []}
    cut = n//cut_factor
    dat_combined = np.column_stack([dat['X'], dat['y']])
    nblocks = n//block_length
    
    blocks = []
    for i in np.arange(n-block_length+1):
        blocks.append(dat_combined[i:i+block_length,:])
        
    samples = [np.random.choice(len(blocks),size=nblocks,replace=True) for _ in range(B)]
    dat_blocks = [dat_combined] + [np.row_stack([blocks[i] for i in sample]) for sample in samples]
    
    Theta = Thetabar = np.zeros([B+1,p])
    for i in range(n):
        alpha_t = alpha0 * (i+1)**(-eta)
        for b in range(B+1):
            X = dat_blocks[b][i,:-1]
            y = dat_blocks[b][i,-1]
            Theta[b,:] += 2*alpha_t*(y - np.dot(Theta[b,:], X))*X
        if i > cut:
            Thetabar = ((i-cut-1)*Thetabar + Theta)/(i-cut)
        else:
            Thetabar = Theta
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

@ray.remote
def run_bootstrap(seed,p,q,n,theta0,states,tpm,stat_dist,B,alpha0,eta,cut_factor=5,block_length=10,freq=10):
    np.random.seed(seed)
    if seed % 50 == 0:
        now = dt.datetime.now()
        print (seed, now.strftime("%Y-%m-%d %H:%M:%S"))
    dat = gen_linear(p,q,n,theta0,states,tpm,stat_dist)
    CIs_online = online_bootstrap(dat, B, alpha0, eta, cut_factor, freq)
    CIs_offline = mb_bootstrap(dat, B, alpha0, eta, cut_factor, freq, block_length)
    return {'online': CIs_online, 'offline': CIs_offline}
    
if __name__ == "__main__":
    num_cpus = psutil.cpu_count(logical=False)
    ray.init(num_cpus=num_cpus)
    
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
    theta0 = np.concatenate((np.repeat(mu,np.round(q/2)), np.repeat(-mu,q-np.round(q/2)), np.repeat(0,p-q)))
    args = {'n': n,'p': p,'q': q,'B': B,'alpha0': alpha0,'eta': eta,'theta0': theta0,
            'states':states,'tpm':tpm,'stat_dist':stat_dist, 'block_length': block_length,
    }
    
    res = ray.get([run_bootstrap.remote(seed, **args) for seed in range(niter)])
    
    online_Q = [r['online']['Q'] for r in res]
    online_SE = [r['online']['SE'] for r in res]
    offline_Q = [r['offline']['Q'] for r in res]
    offline_SE = [r['offline']['SE'] for r in res]
    
    dim = 0
    run = 5
    start=100
    
    def widths(X, dim, run):
        return [CIs[dim,2] - CIs[dim,0] for CIs in X[run]]
        
    x = np.arange(len(widths(online_Q,0,0)))*10

    plt.rcParams['figure.figsize'] = (16,9)
    plt.rcParams['figure.facecolor'] = 'white'
    fig,axs = plt.subplots(2,2)

    dim=0
    axs[0,0].plot(x[start:], widths(online_SE,dim,run)[start:], label='online')
    axs[0,0].plot(x[start:], widths(offline_SE,dim,run)[start:], label='offline', color='red')
    axs[0,0].set(ylabel='CI Width', title='Dimension {}'.format(dim))
    axs[0,0].legend()

    dim=3
    axs[0,1].plot(x[start:], widths(online_SE,dim,run)[start:], label='online')
    axs[0,1].plot(x[start:], widths(offline_SE,dim,run)[start:], label='offline', color='red')
    axs[0,1].set(ylabel='CI Width', title='Dimension {}'.format(dim))
    axs[0,1].legend()

    dim=6
    axs[1,0].plot(x[start:], widths(online_SE,dim,run)[start:], label='online')
    axs[1,0].plot(x[start:], widths(offline_SE,dim,run)[start:], label='offline', color='red')
    axs[1,0].set(xlabel='Number of observations', ylabel='CI Width', title='Dimension {}'.format(dim))
    axs[1,0].legend()

    dim=9
    axs[1,1].plot(x[start:], widths(online_SE,dim,run)[start:], label='online')
    axs[1,1].plot(x[start:], widths(offline_SE,dim,run)[start:], label='offline', color='red')
    axs[1,1].set(xlabel='Number of observations', ylabel='CI Width', title='Dimension {}'.format(dim))
    axs[1,1].legend()

    plt.savefig('markov_rw_CI_width.png', bbox_inches='tight')
    plt.close()
    
    start=10
    def coverage(X, theta0, dim):
        return np.array([np.array([(theta0[dim] > CIs[dim,0]) & (theta0[dim] < CIs[dim,2]) for CIs in r]) for r in X]).mean(axis=0)
        
    plt.rcParams['figure.figsize'] = (16,9)
    plt.rcParams['figure.facecolor'] = 'white'
    fig,axs = plt.subplots(2,2)
    start=10

    dim=0
    axs[0,0].plot(x[start:], coverage(online_SE,theta0, dim)[start:], label='online')
    axs[0,0].plot(x[start:], coverage(offline_SE,theta0, dim)[start:], label='offline', color='red')
    axs[0,0].axhline(0.95, color='black', linestyle='--')
    axs[0,0].set(ylabel='Coverage', title='Dimension {}'.format(dim), ylim=(0.8,1.01))
    axs[0,0].legend(loc='lower right')

    dim=3
    axs[0,1].plot(x[start:], coverage(online_SE,theta0, dim)[start:], label='online')
    axs[0,1].plot(x[start:], coverage(offline_SE,theta0, dim)[start:], label='offline', color='red')
    axs[0,1].axhline(0.95, color='black', linestyle='--')
    axs[0,1].set(ylabel='Coverage', title='Dimension {}'.format(dim), ylim=(0.8,1.01))
    axs[0,1].legend(loc='lower right')

    dim=6
    axs[1,0].plot(x[start:], coverage(online_SE,theta0, dim)[start:], label='online')
    axs[1,0].plot(x[start:], coverage(offline_SE,theta0, dim)[start:], label='offline', color='red')
    axs[1,0].axhline(0.95, color='black', linestyle='--')
    axs[1,0].set(ylabel='Coverage', title='Dimension {}'.format(dim), xlabel='Number of observations', ylim=(0.8,1.01))
    axs[1,0].legend(loc='lower right')

    dim=9
    axs[1,1].plot(x[start:], coverage(online_SE,theta0, dim)[start:], label='online')
    axs[1,1].plot(x[start:], coverage(offline_SE,theta0, dim)[start:], label='offline', color='red')
    axs[1,1].axhline(0.95, color='black', linestyle='--')
    axs[1,1].set(ylabel='Coverage', title='Dimension {}'.format(dim), xlabel='Number of observations', ylim=(0.8,1.01))
    axs[1,1].legend(loc='lower right')

    plt.savefig('markov_rw_CI_coverage.png', bbox_inches='tight')
    plt.close()