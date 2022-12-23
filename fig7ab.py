import numpy as np
from sepsisSimDiabetes.MDP import MDP
from sepsisSimDiabetes.State import State
from sepsisSimDiabetes.Action import Action
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def gen_data(mdp, policy, n_episodes):
    Data = []
    num_actions = Action.NUM_ACTIONS_TOTAL
    for i in tqdm(range(n_episodes)):
        data = []
        done=False
        state=mdp.get_new_state()
        while not done:
            action = np.random.choice(num_actions, p=policy[state.get_state_idx(),:])
            reward = mdp.transition(Action(action_idx=action))
            state_ = mdp.state
            data.append((state.get_state_idx(),action,reward,state_.get_state_idx()))
            state = state_
            done = mdp.state.check_absorbing_state()
        Data.append(data)
    return Data

def gtdlearn_NEU(Data, Phi, policy_target, policy_behavior, gamma, alpha0, eta):
    NEU = []
    p = 7
    theta = np.zeros([2*p])
    thetabar = np.zeros(p)
    n_episodes = len(Data)
    cut = n_episodes//cut_factor
    num_actions = Action.NUM_ACTIONS_TOTAL
    
    j=0
    for i in tqdm(range(n_episodes)):
        data = Data[i]
        for k in range(len(data)):
            j+=1
            alpha_t = alpha0 * j**(-eta)
            state,action,reward,state_ = data[k]
            phi = Phi[state,:]
            phi_ = Phi[state_,:]
            rho_t = policy_target[state,action]/policy_behavior[state,action]
            At = rho_t*phi[:,np.newaxis]@(phi - gamma*phi_)[np.newaxis,:]
            bt = rho_t*reward*phi
            Mt = np.eye(p)
            b1t = np.concatenate([np.zeros(p),bt])
            A1t = np.row_stack([np.column_stack([np.zeros([p,p]), -At.T]), np.column_stack([At,Mt])])
            theta += alpha_t*(b1t - A1t@theta)
            if j > cut:
                thetabar = ((j-cut-1)*thetabar + theta[:p])/(j-cut)
                delta_t = reward + np.dot((gamma*phi_ - phi), thetabar[:p])
                NEU.append([delta_t**2 * np.dot(phi,phi)])
            else:
                thetabar=theta[:p]
    NEU = np.array(NEU)
    return np.cumsum(NEU) / np.arange(1,len(NEU)+1)

def online_bootstrap(Data, Phi, policy_target, policy_behavior, gamma, alpha0, eta, B, cut_factor=1, freq=10):
    CIs = {'Q':[], 'SE':[]}
    p = 7
    Theta = np.zeros([B+1,2*p])
    Thetabar = np.zeros([B+1,p])
    n_episodes = len(Data)
    cut = n_episodes//cut_factor
    num_actions = Action.NUM_ACTIONS_TOTAL
    
    j=0
    for i in tqdm(range(n_episodes)):
        data = Data[i]
        for k in range(len(data)):
            j+=1
            alpha_t = alpha0 * j**(-eta)
            W = np.concatenate([[1], np.random.exponential(size=B)])
            state,action,reward,state_ = data[k]
            phi = Phi[state,:]
            phi_ = Phi[state_,:]
            rho_t = policy_target[state,action]/policy_behavior[state,action]
            At = rho_t*phi[:,np.newaxis]@(phi - gamma*phi_)[np.newaxis,:]
            bt = rho_t*reward*phi
            Mt = np.eye(p)
            b1t = np.concatenate([np.zeros(p),bt])
            A1t = np.row_stack([np.column_stack([np.zeros([p,p]), -At.T]), np.column_stack([At,Mt])])
            Theta += alpha_t*np.diag(W) @ (b1t[np.newaxis,:] - Theta@A1t.T)
            if j > cut:
                Thetabar = ((j-cut-1)*Thetabar + Theta[:,:p])/(j-cut)
            else:
                Thetabar=Theta[:,:p]
        if i % freq == 0:
            thetabar = Thetabar[0,:]
            ThetabarW = Thetabar[1:,:]
            values = Phi@Thetabar.T
            value0 = values[:,0]
            valueW = values[:,1:]
            CI_Q = value0[:,np.newaxis] + np.quantile(value0[:,np.newaxis] - valueW, [0.025,0.975], axis=1).T
            SE = np.sqrt(np.diag(Phi @ np.cov(thetabar[np.newaxis,:] - ThetabarW, rowvar=False) @ Phi.T))
            CI_SE = value0[:,np.newaxis] + 1.96*np.outer(SE, [-1,1])
            CI_SE = np.insert(CI_SE,1,value0,axis=1)
            CI_Q = np.insert(CI_Q,1,value0,axis=1)
            CIs['Q'].append(CI_Q)
            CIs['SE'].append(CI_SE)
    return CIs

if __name__ == "__main__":
    num_actions = Action.NUM_ACTIONS_TOTAL
    num_states = State.NUM_OBS_STATES
    policy = np.ones((num_states, num_actions)) / num_actions
    mdp = MDP(policy_array=policy, p_diabetes=0)
    p=7

    Q_table = np.loadtxt(os.path.join(os.getcwd(), "Q_gumbel.csv"), delimiter=",")
    policy_Q = np.array([np.eye(num_actions)[np.argmax(row),:] if row.sum() > 0 else np.ones(num_actions)/num_actions for row in Q_table])

    epsilon = 0.25
    policy = (1-epsilon)*policy_Q + epsilon*np.ones((num_states, num_actions)) / num_actions

    Phi = np.zeros([num_states,p])
    for i in range(num_states):
        Phi[i,:] = State(state_idx=i,diabetic_idx = 0).get_state_vector()
        
    # params
    n_episodes = int(5e4)
    alpha0 = 0.1
    eta = 3/4
    gamma = 0.99
    B = 200
    cut_factor=20

    Data = gen_data(mdp, policy, n_episodes)

    CIs = online_bootstrap(Data, Phi, policy_Q, policy, gamma, alpha0, eta, B, cut_factor=cut_factor)

    NEU = gtdlearn_NEU(Data, Phi, policy_Q, policy, gamma, alpha0, eta)

    # fig7a
    NEU_resample = NEU[::(len(NEU)//n_episodes)]
    start = 5000
    plt.rcParams['figure.figsize'] = (16,9)
    plt.rcParams['figure.facecolor'] = 'white'
    plt.plot(np.arange(len(NEU_resample))[start:],np.log(NEU_resample)[start:], linewidth=2)
    plt.xlabel('Number of episodes', fontsize=20)
    plt.ylabel('log(NEU)', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig('gumbel_NEU.png', bbox_inches='tight')
    plt.close()

    # fig7b
    dim = 1
    CIs_Q = np.array([CI[dim,:] for CI in CIs['Q']])
    CIs_SE = np.array([CI[dim,:] for CI in CIs['SE']])
    CIs_Q_width = CIs_Q[:,2] - CIs_Q[:,0]
    CIs_SE_width = CIs_SE[:,2] - CIs_SE[:,0]
    start = 100
    x = (np.arange(len(CIs_Q_width))*10)[start:]
    plt.rcParams['figure.figsize'] = (16,9)
    plt.rcParams['figure.facecolor'] = 'white'
    plt.plot(x, CIs_Q_width[start:], label='Q', color='blue', linewidth=2)
    plt.plot(x, CIs_SE_width[start:], label = 'SE', color='red', linewidth=2)
    plt.xlabel('Number of episodes', fontsize=20)
    plt.ylabel('CI Width', fontsize=20)
    plt.legend(fontsize='xx-large')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig('gumbel_CIwidth.png', bbox_inches='tight')
    plt.close()