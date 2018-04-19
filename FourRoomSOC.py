import gym
import argparse
import numpy as np
from fourrooms import Fourrooms
import math
from scipy.special import expit
from scipy.misc import logsumexp
import os
import datetime


class Tabular:
    def __init__(self, nstates):
        self.nstates = nstates

    def __call__(self, state):
        return np.array([state,])

    def __len__(self):
        return self.nstates

class EgreedyPolicy:
    def __init__(self, rng, nfeatures, nactions, epsilon):
        self.rng = rng
        self.epsilon = epsilon
        self.nactions = nactions
        self.weights = 0.5*np.ones((nfeatures, nactions))

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def sample(self, phi, curr_epsilon):
        if self.rng.uniform() < curr_epsilon:
            return int(self.rng.randint(self.nactions))
        return int(np.argmax(self.value(phi)))

class SoftmaxPolicy:
    def __init__(self, rng, nfeatures, nactions, temp=1.):
        self.rng = rng
        self.weights = 0.5*np.ones((nfeatures, nactions))
        self.nactions = nactions
        self.temp = temp

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def pmf(self, phi):
        v = self.value(phi)/self.temp
        return np.exp(v - logsumexp(v))

    def sample(self, phi):
        return int(self.rng.choice(self.nactions, p=self.pmf(phi)))

class SigmoidTermination:
    def __init__(self, rng, nfeatures):
        self.rng = rng
        self.weights = 0.5*np.ones((nfeatures,))

    def pmf(self, phi):
        return expit(np.sum(self.weights[phi]))

    def sample(self, phi):
        return int(self.rng.uniform() < self.pmf(phi))

    def grad(self, phi):
        terminate = self.pmf(phi)
        return terminate*(1. - terminate), phi

class IntraOptionQLearning:
    def __init__(self, discount, lr, terminations, weights):
        self.lr = lr
        self.discount = discount
        self.terminations = terminations
        self.weights = weights

    def start(self, phi, option):
        self.last_phi = phi
        self.last_option = option
        self.last_value = self.value(phi, option)

    def value(self, phi, option=None):
        if option is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, option], axis=0)

    def advantage(self, phi, option=None):
        values = self.value(phi)
        advantages = values - np.max(values)
        if option is None:
            return advantages
        return advantages[option]
    #
    def update(self, phi, option, reward, done):
        # One-step update target
        update_target = reward
        if not done:
            current_values = self.value(phi)
            termination = self.terminations[self.last_option].pmf(phi)
            update_target += self.discount*((1. - termination)*current_values[self.last_option] + termination*np.max(current_values))

        # Dense gradient update step
        tderror = update_target - self.last_value
        self.weights[self.last_phi, self.last_option] += self.lr*tderror

        if not done:
            self.last_value = current_values[option]
        self.last_option = option
        self.last_phi = phi

        return update_target

class IntraOptionActionQLearning:
    def __init__(self, discount, lr, terminations, weights, qbigomega):
        self.lr = lr
        self.discount = discount
        self.terminations = terminations
        self.weights = weights
        self.qbigomega = qbigomega


    def value(self, phi, option, action):
        return np.sum(self.weights[phi, option, action], axis=0)

    def start(self, phi, option, action):
        self.last_phi = phi
        self.last_option = option
        self.last_action = action
        
    #
    def update(self, phi, option, action, reward, done):
        # One-step update target
        update_target = reward
        if not done:
            current_values = self.qbigomega.value(phi)
            termination = self.terminations[self.last_option].pmf(phi)
            update_target += self.discount*((1. - termination)*current_values[self.last_option] + termination*np.max(current_values))

        # Update values upon arrival if desired
        tderror = update_target - self.value(self.last_phi, self.last_option, self.last_action)
        self.weights[self.last_phi, self.last_option, self.last_action] += self.lr*tderror

        self.last_phi = phi
        self.last_option = option
        self.last_action = action

    def get_initial_td_error(self, initial_phi, initial_option, action, reward, next_phi, beta):
        termination = self.terminations[initial_option].pmf(next_phi)
        val_next_state = self.qbigomega.value(next_phi)
        initial_td_error = reward + self.discount*((1. - termination)* val_next_state[initial_option]\
         + termination * np.max(val_next_state)) - self.value(initial_phi, initial_option, action)

        return beta * math.pow(initial_td_error, 2.0)

class TerminationGradient:
    def __init__(self, terminations, critic, lr):
        self.terminations = terminations
        self.critic = critic
        self.lr = lr
    
    def update(self, phi, option):
        magnitude, direction = self.terminations[option].grad(phi)
        self.terminations[option].weights[direction] -= \
                self.lr*magnitude*(self.critic.advantage(phi, option))

class IntraOptionGradient:
    def __init__(self, option_policies, lr, beta, nactions):
        self.lr = lr
        self.option_policies = option_policies
        self.beta = beta
        self.initial_actions_pmf = np.zeros(nactions)
        self.visit_phi = 1
        self.val_subtract = 0.0
        self.val_add = 0.0

    def update(self, phi, option, action, critic, initial_td_error, initial_option, initial_phi, initial_action, controllability = False):
        actions_pmf = self.option_policies[option].pmf(phi)
        if controllability == True:
            if self.visit_phi > 0:
                self.initial_actions_pmf = self.option_policies[initial_option].pmf(initial_phi)
                self.val_subtract = self.lr*initial_td_error
                self.val_add = self.val_subtract*self.initial_actions_pmf
                self.visit_phi = 0
            
            self.option_policies[initial_option].weights[initial_phi, initial_action] -= self.val_subtract
            self.option_policies[initial_option].weights[initial_phi, :] += self.val_add
        
        curr_val = self.lr*critic
        self.option_policies[option].weights[phi, :] -= curr_val*actions_pmf
        self.option_policies[option].weights[phi, action] += curr_val

        if (controllability == True) and (initial_option == option):
            if np.intersect1d(phi, initial_phi).size > 0:
                self.visit_phi = 1

class OneStepTermination:
    def sample(self, phi):
        return 1

    def pmf(self, phi):
        return 1.

class FixedActionPolicies:
    def __init__(self, action, nactions):
        self.action = action
        self.probs = np.eye(nactions)[action]

    def sample(self, phi):
        return self.action

    def pmf(self, phi):
        return self.probs


def GetFrozenStates():
    layout ="""\
wwwwwwwwwwwww
w     w     w
w   ffwff   w
w  fffffff  w
w   ffwff   w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""

    num_elem = 13
    line_count = 0
    element_count = 0
    frozen_states =[]
    state_num=0
    for line in layout.splitlines():
        for i in range(num_elem):        
            if line[i] == "f":
                frozen_states.append(state_num)
            if line[i]!="w":
                state_num +=1
    return frozen_states

def save_params(args, dir_name):
    f = os.path.join(dir_name, "Params.txt")
    with open(f, "w") as f_w:
        for param, val in sorted(vars(args).items()):
            f_w.write("{0}:{1}\n".format(param, val))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--discount', help='Discount factor', type=float, default=0.99)
    parser.add_argument('--lr_term', help="Termination gradient learning rate", type=float, default=1e-1)
    parser.add_argument('--lr_intra', help="Intra-option gradient learning rate", type=float, default=1e-2)
    parser.add_argument('--lr_critic', help="Learning rate", type=float, default=5e-1)
    parser.add_argument('--epsilon', help="Epsilon-greedy for policy over options", type=float, default=2e-1)
    parser.add_argument('--nepisodes', help="Number of episodes per run", type=int, default= 2000)
    parser.add_argument('--nruns', help="Number of runs", type=int, default=50)
    parser.add_argument('--nsteps', help="Maximum number of steps per episode", type=int, default=500)
    parser.add_argument('--noptions', help='Number of options', type=int, default=4)
    parser.add_argument('--baseline', help="Use the baseline for the intra-option gradient", action='store_true', default=False)
    parser.add_argument('--beta', help="Beta for controllability", type=float, default=0.05)
    parser.add_argument('--temperature', help="Temperature parameter for softmax", type=float, default=1e-3)
    parser.add_argument('--controllability', help="whether controllability experiment", type=str2bool, default=True)
    parser.add_argument('--policyOverOptions', help="softmax or e-greedy", type=str2bool, default=False) #False: Softmax Policy, True: E-greedy Policy
    parser.add_argument('--intraOptionPolicy', help="softmax or e-greedy", type=str2bool, default=False) #False: Softmax Policy, True: E-greedy Policy
    parser.add_argument('--epsilondelay', help="seed value for experiment", type=int, default=500)
    parser.add_argument('--seed', help="seed value for experiment", type=int, default=10)
    
    
    args = parser.parse_args()
    rng = np.random.RandomState(args.seed)
    env = gym.make('Fourrooms-v0')

    outer_dir = "FourRoomOption"
    if not os.path.exists(outer_dir):
        os.makedirs(outer_dir)

    dir_name = "R"+str(args.nruns)+"_E"+str(args.nepisodes)+"_Beta"+str(args.beta) +\
     "_Eps"+str(args.epsilon)+"_nopt"+str(args.noptions)+"_LRT"+ str(args.lr_term) +\
     "_LRI"+str(args.lr_intra)+"_LRC"+str(args.lr_critic)+"_temp"+str(args.temperature)+"_seed"+str(args.seed)
    
    if args.policyOverOptions== True:
        dir_name += "_Policy_" + "G"
    else:
        dir_name += "_Policy_" + "S"

    if args.intraOptionPolicy== True:
        dir_name += "_IntraOption_" + "G"
    else:
        dir_name += "_IntraOption_" + "S"

    if args.controllability== True:
        dir_name += "_C"

    dir_name = os.path.join(outer_dir, dir_name)
    if not os.path.exists(dir_name):
    	os.makedirs(dir_name)

    save_params(args, dir_name)

    num_states = env.observation_space.n
    num_actions = env.action_space.n
    
    history = np.zeros((args.nruns, args.nepisodes, 2), dtype=np.float32)
    frozen_states = GetFrozenStates()

    state_frequency_history = np.zeros((args.nruns, args.nepisodes, env.observation_space.n, args.noptions),dtype=np.int32)
    

    # for storing the weights of the trained model
    weight_intra_option = np.zeros((args.nruns ,args.nepisodes, num_states, num_actions,  args.noptions), dtype=np.float32)
    weight_policy = np.zeros((args.nruns ,args.nepisodes, num_states, args.noptions), dtype=np.float32)
    weight_termination = np.zeros((args.nruns ,args.nepisodes, num_states, args.noptions), dtype=np.float32)

    #decaying epsilon which settles at value specified in params
    decay_rate = args.epsilon / float(args.epsilondelay)
    for run in range(args.nruns):
        features = Tabular(env.observation_space.n)
        nfeatures, nactions = len(features), env.action_space.n

        # The intra-option policies are linear-softmax functions

        if args.intraOptionPolicy == True:
            option_policies = [EgreedyPolicy(rng, nfeatures, nactions, args.temperature) for _ in range(args.noptions)]
        else:
            option_policies = [SoftmaxPolicy(rng, nfeatures, nactions, args.temperature) for _ in range(args.noptions)]

        # The termination function are linear-sigmoid functions
        option_terminations = [SigmoidTermination(rng, nfeatures) for _ in range(args.noptions)]

        # E-greedy policy over options
        if args.policyOverOptions == True:
            policy = EgreedyPolicy(rng, nfeatures, args.noptions, args.epsilon)
        else:
            policy = SoftmaxPolicy(rng, nfeatures, args.noptions, args.temperature)

        # Different choices are possible for the critic. Here we learn an
        # option-value function and use the estimator for the values upon arrival
        critic = IntraOptionQLearning(args.discount, args.lr_critic, option_terminations, policy.weights)

        # Learn Qomega separately
        action_weights = np.random.random((nfeatures, args.noptions, nactions))
        action_critic = IntraOptionActionQLearning(args.discount, args.lr_critic, option_terminations, action_weights, critic)

        # Improvement of the termination functions based on gradients
        termination_improvement= TerminationGradient(option_terminations, critic, args.lr_term)

        # Intra-option gradient improvement with critic estimator
        intraoption_improvement = IntraOptionGradient(option_policies, args.lr_intra, args.beta, num_actions)
        
        curr_epsilon = 1.0

        for episode in range(args.nepisodes):
            return_per_episode = 0
            observation = env.reset()
            phi = features(observation)
            curr_epsilon = max(curr_epsilon - decay_rate, args.epsilon)

            if args.policyOverOptions== True: #Greedy
                option = policy.sample(phi, curr_epsilon)
            else:
                option = policy.sample(phi)

            action = option_policies[option].sample(phi)
            critic.start(phi, option)
            action_critic.start(phi, option, action)

            initial_phi = phi
            initial_action = action
            initial_option = option
            intraoption_improvement.visit_phi = 1

            intial_td_error = 0.0
            for step in range(args.nsteps):
            	old_phi = phi
            	old_option = option
            	old_action = action
                observation, reward, done, _ = env.step(action)
                
                #if episode >= episode_which_to_save_state_freq:
                state_frequency_history[run, episode, observation, option] +=1

                #Frozen state receives a variable uniform reward[-15, 15]
                if observation in frozen_states:
                    reward = np.random.uniform(-15.0, 15.0)
               
                phi = features(observation)
                #return calculation
                return_per_episode += pow(args.discount,step)*reward

                # Termination might occur upon entering the new state
                if option_terminations[option].sample(phi):
                    if args.policyOverOptions== True:
                        option = policy.sample(phi, curr_epsilon)
                    else:
                        option = policy.sample(phi)
               
                action = option_policies[option].sample(phi)

                # Critic update
                update_target = critic.update(phi, option, reward, done)
                action_critic.update(phi, option, action, reward, done)

                if ((args.controllability ==True) and (old_phi == initial_phi and old_option == initial_option)):
                    second_phi = phi
                    initial_reward = reward
                    initial_action = old_action
                    intial_td_error = action_critic.get_initial_td_error(initial_phi, initial_option, initial_action,
                     initial_reward, second_phi, args.beta)

                if isinstance(option_policies[option], SoftmaxPolicy):
                    # Intra-option policy update
                    critic_feedback = action_critic.value(old_phi, old_option, old_action)
                    if args.baseline:
                        critic_feedback -= critic.value(old_phi, old_option)

                    intraoption_improvement.update(old_phi, old_option, old_action, critic_feedback, intial_td_error,
                        initial_option, initial_phi, initial_action, controllability = args.controllability)
                    termination_improvement.update(phi, old_option)

                if done:
                    break
            
            history[run, episode, 0] = step
            history[run, episode, 1] = return_per_episode

            
            for o in range(args.noptions):
                weight_intra_option[run,episode, :, :, o] = option_policies[o].weights  
            weight_policy[run,episode , :, :] = policy.weights
            for o in range(args.noptions):
                weight_termination[run,episode , :, o] = option_terminations[o].weights       

    np.save(os.path.join(dir_name,'stateFreq.npy'), state_frequency_history)
    np.save(os.path.join(dir_name,'Weights_Policy.npy'), weight_policy)
    np.save(os.path.join(dir_name,'Weights_Termination.npy'), weight_termination)
    np.save(os.path.join(dir_name,'Weights_IntraOption.npy'), weight_intra_option)
    np.save(os.path.join(dir_name,'History.npy'), history)