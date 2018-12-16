import gym
import argparse
import numpy as np
import math
from scipy.special import expit
from scipy.misc import logsumexp
import os
from tiles3 import *
import threading
import datetime
from puddlesimple import PuddleSimpleEnv

class TileFeature:
    def __init__(self, ntiles, nbins, discrete_states, features_range):
        self.ntiles = ntiles
        self.nbins = nbins
        self.max_discrete_states = discrete_states
        self.iht = IHT(discrete_states)
        self.features_range = features_range
        self.scaling = nbins /features_range

    def __call__(self, input_observation):
        return tiles(self.iht, self.ntiles, input_observation*self.scaling)

    def __len__(self):
        return self.max_discrete_states

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

class BoltzmannPolicy:
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
    def __init__(self, gamma, lr, terminations, weights):
        self.lr = lr
        self.gamma = gamma
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

    def update(self, phi, option, reward, done):
        # One-step update target
        update_target = reward
        if not done:
            current_values = self.value(phi)
            termination = self.terminations[self.last_option].pmf(phi)
            update_target += self.gamma *((1. - termination)*current_values[self.last_option] + termination*np.max(current_values))

        # Dense gradient update step
        tderror = update_target - self.last_value
        self.weights[self.last_phi, self.last_option] += self.lr*tderror
        if not done:
            self.last_value = current_values[option]
        self.last_option = option
        self.last_phi = phi

        return update_target

class IntraOptionActionQLearning:
    def __init__(self, gamma, lr, terminations, weights, qbigomega):
        self.lr = lr
        self.gamma = gamma
        self.terminations = terminations
        self.weights = weights
        self.qbigomega = qbigomega


    def value(self, phi, option, action):
        return np.sum(self.weights[phi, option, action], axis=0)

    def start(self, phi, option, action):
        self.last_phi = phi
        self.last_option = option
        self.last_action = action

    def update(self, phi, option, action, reward, done):
        # One-step update target
        update_target = reward
        if not done:
            current_values = self.qbigomega.value(phi)
            termination = self.terminations[self.last_option].pmf(phi)
            update_target += self.gamma*((1. - termination)*current_values[self.last_option] + termination*np.max(current_values))

        # Update values upon arrival if desired
        tderror = update_target - self.value(self.last_phi, self.last_option, self.last_action)
        self.weights[self.last_phi, self.last_option, self.last_action] += self.lr*tderror

        self.last_phi = phi
        self.last_option = option
        self.last_action = action
        return tderror

    def get_initial_td_error(self, initial_phi, initial_option, action, reward, next_phi, psi):
        termination = self.terminations[initial_option].pmf(next_phi)
        val_next_state = self.qbigomega.value(next_phi)
        initial_td_error = reward + self.gamma*((1. - termination)* val_next_state[initial_option]\
         + termination * np.max(val_next_state)) - self.value(initial_phi, initial_option, action)

        return psi * math.pow(initial_td_error, 2.0)

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
    def __init__(self, option_policies, lr, psi, nactions):
        self.lr = lr
        self.option_policies = option_policies
        self.psi = psi
        self.initial_actions_pmf = np.zeros(nactions)
        self.visit_phi = 1
        self.val_subtract = 0.0
        self.val_add = 0.0

    def update(self, phi, option, action, critic, initial_td_error, initial_option, initial_phi, initial_action):
        actions_pmf = self.option_policies[option].pmf(phi)
        if self.psi!=0.0:
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

        if (self.psi!=0.0) and (initial_option == option):
            if np.intersect1d(phi, initial_phi).size > 0:
                self.visit_phi = 1


class OutputInformation:
    def __init__(self):
        # storaging the weights of the trained model
        self.weight_intra_option = []
        self.weight_policy = []
        self.weight_termination = []
        self.history = []
        self.td_states = []
        self.visit_states = []

def tweak_reward_near_puddle(observation, reward):
    noise_mean = 0.0
    noise_sigma = 8.0
    if (check_if_agent_near_puddle(observation)):
        noise = np.random.normal(noise_mean, noise_sigma)
        return reward + noise
    return reward


# This function checks whether the agent has entered puddle zone
def check_if_agent_near_puddle(observation):
    if (observation[0] <= 0.7 and observation[0] >= 0.3):
        if (observation[1] <= 0.7 and observation[1] >= 0.3):
            return True
    return False

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

def run_agent(outputinfo, nepisodes, nfeatures, nactions, num_states,
              temperature, gamma, lr_critic, lr_intra, lr_term, psi,
              noptions, ntiles, nbins, maxDiscreteStates, rng):

    after_every_episode =100
    episode_now = int(nepisodes/after_every_episode)
    history = np.zeros((nepisodes, 3), dtype=np.float32)  # 1. Return 2. Steps 3. TD error 1 norm

    # store the weights of the trained model
    weight_intra_option = np.zeros((episode_now, num_states, nactions,  noptions), dtype=np.float32)
    weight_policy = np.zeros((episode_now, num_states, noptions), dtype=np.float32)
    weight_termination = np.zeros((episode_now, num_states, noptions), dtype=np.float32)

    # Intra-Option Policy
    option_policies = [BoltzmannPolicy(rng, nfeatures, nactions, temperature) for _ in range(noptions)]
    # The termination function are linear-sigmoid functions
    option_terminations = [SigmoidTermination(rng, nfeatures) for _ in range(noptions)]
    # Policy over options
    policy = BoltzmannPolicy(rng, nfeatures, noptions, temperature)

    # Different choices are possible for the critic. Here we learn an
    # option-value function and use the estimator for the values upon arrival
    critic = IntraOptionQLearning(gamma, lr_critic, option_terminations, policy.weights)

    # Learn Qomega separately
    action_weights = np.ones((nfeatures, noptions, nactions))*0.5
    action_critic = IntraOptionActionQLearning(gamma, lr_critic, option_terminations, action_weights, critic)

    # Improvement of the termination functions based on gradients
    termination_improvement= TerminationGradient(option_terminations, critic, lr_term)

    # Intra-option gradient improvement with critic estimator
    intraoption_improvement = IntraOptionGradient(option_policies, lr_intra, psi, nactions)

    env = gym.make('PuddleEnv-v0')
    features_range = env.observation_space.high - env.observation_space.low
    features = TileFeature(ntiles, nbins, maxDiscreteStates, features_range)
    run_td =[]
    run_visit =[]


    for episode in range(nepisodes):
        episode_td =[]
        episode_visit =[]

        return_per_episode = 0
        observation = env.reset()
        phi = features(observation)
        option = policy.sample(phi)
        action = option_policies[option].sample(phi)
        critic.start(phi, option)
        action_critic.start(phi, option, action)
        initial_phi = phi
        initial_action = action
        initial_option = option
        intraoption_improvement.visit_phi = 1

        intial_td_error = 0.0
        done = False
        step = 0
        sum_td_error = 0.0
        while done!=True:
            old_observation = observation
            old_phi = phi
            old_option = option
            old_action = action
            observation, reward, done, _ = env.step(action)

            phi = features(observation)
            if check_if_agent_near_puddle(old_observation):
                reward = tweak_reward_near_puddle(old_observation, reward)
            return_per_episode += pow(gamma,step)*reward
    
        # Termination might occur upon entering the new state
            if option_terminations[option].sample(phi):
               option = policy.sample(phi)

            action = option_policies[option].sample(phi)

            # Critic update
            update_target = critic.update(phi, option, reward, done)
            tderror = action_critic.update(phi, option, action, reward, done)
            sum_td_error += abs(tderror)

            if ((psi != 0.0 ) and (old_phi == initial_phi and old_option == initial_option)):
                second_phi = phi
                initial_reward = reward
                initial_action = old_action
                intial_td_error = action_critic.get_initial_td_error(initial_phi, initial_option, initial_action,
                 initial_reward, second_phi, psi)

            # Intra-option policy update
            critic_feedback = action_critic.value(old_phi, old_option, old_action) - critic.value(old_phi, old_option)
            intraoption_improvement.update(old_phi, old_option, old_action, critic_feedback, intial_td_error,
                initial_option, initial_phi, initial_action)
            termination_improvement.update(phi, old_option)
            td_states = []
            td_states.extend(old_observation)
            td_states.extend([tderror])

            visit_states =[]
            visit_states.extend(old_observation)
            visit_states.extend([old_option])

            episode_td.append(td_states)
            episode_visit.append(visit_states)
            step +=1
            if done:
                break

        history[episode, 0] = step
        history[episode, 1] = return_per_episode
        history[episode, 2] = sum_td_error


        if episode % after_every_episode ==0:
            this_episode = int(episode/after_every_episode)

            for o in range(noptions):
                weight_intra_option[this_episode, :, :, o] = option_policies[o].weights
            weight_policy[this_episode , :, :] = policy.weights
            for o in range(noptions):
                weight_termination[this_episode , :, o] = option_terminations[o].weights

        run_td.append(episode_td)
        run_visit.append(episode_visit)

    outputinfo.weight_intra_option.append(weight_intra_option)
    outputinfo.weight_policy.append(weight_policy)
    outputinfo.weight_termination.append(weight_termination)
    outputinfo.history.append(history)
    outputinfo.td_states.append(run_td)
    outputinfo.visit_states.append(run_visit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', help='gamma factor Gamma', type=float, default=0.99)
    parser.add_argument('--lr_term', help="Termination gradient learning rate", type=float, default=0.1)
    parser.add_argument('--lr_intra', help="Intra-option gradient learning rate", type=float, default=0.1)
    parser.add_argument('--lr_critic', help="Learning rate", type=float, default=0.1)
    parser.add_argument('--nepisodes', help="Number of episodes per run", type=int, default= 1000)
    parser.add_argument('--nruns', help="Number of runs", type=int, default=1)
    parser.add_argument('--noptions', help='Number of options', type=int, default=2)
    parser.add_argument('--psi', help="psi for controllability", type=float, default=0.0)
    parser.add_argument('--temperature', help="Temperature parameter for Boltzmann", type=float, default=0.5)
    parser.add_argument('--maxDiscreteStates', help="num states to quantize continuous state", type=int, default=1024)
    parser.add_argument('--ntiles', help="num of tilings", type=int, default=5)
    parser.add_argument('--nbins', help="binning in tile coding", type=int, default=5)
    parser.add_argument('--seed', help="seed value for experiment", type=int, default=20)

    args = parser.parse_args()
    now_time = datetime.datetime.now()
    env = gym.make('PuddleEnv-v0')
    features_range = env.observation_space.high - env.observation_space.low
    features = TileFeature(args.ntiles, args.nbins, args.maxDiscreteStates, features_range)

    outer_dir = "results"
    if not os.path.exists(outer_dir):
        os.makedirs(outer_dir)
    outer_dir = os.path.join(outer_dir, "PuddleNew_" + now_time.strftime("%d-%m"))
    if not os.path.exists(outer_dir):
        os.makedirs(outer_dir)
    dir_name = "R" + str(args.nruns) + "_E" + str(args.nepisodes) + "_Psi" + str(args.psi) + \
               "_LRC" + str(args.lr_critic) + "_LRIntra" + str(args.lr_intra) + "_LRT" + str(args.lr_term) + \
               "_nOpt" + str(args.noptions) + "_temp" + str(args.temperature) + "ntile"+ str(args.ntiles)+ "nbins"+ str(args.nbins) +"_seed" + str(args.seed)
    dir_name = os.path.join(outer_dir, dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    save_params(args, dir_name)
    nactions = env.action_space.n
    threads = []
    nfeatures = len(features)
    outputinfo = OutputInformation()
    args.lr_critic /= args.ntiles
    args.lr_intra /= args.ntiles
    args.lr_term /= args.ntiles

    for i in range(args.nruns):
        t = threading.Thread(target=run_agent, args=(outputinfo, args.nepisodes, nfeatures,
                                                     nactions, nfeatures, args.temperature,
                                                     args.gamma, args.lr_critic, args.lr_intra, args.lr_term, args.psi,
                                                     args.noptions, args.ntiles, args.nbins,args.maxDiscreteStates,
                                                     np.random.RandomState(args.seed + i), ))
        threads.append(t)
        t.start()

    for x in threads:
        x.join()

    np.save(os.path.join(dir_name, 'WeightIntraOption.npy'), np.asarray(outputinfo.weight_intra_option))
    np.save(os.path.join(dir_name, 'WeightPolicy.npy'), np.asarray(outputinfo.weight_policy))
    np.save(os.path.join(dir_name, 'WeightTermination.npy'), np.asarray(outputinfo.weight_termination))
    np.save(os.path.join(dir_name, 'History.npy'), np.asarray(outputinfo.history))
    np.save(os.path.join(dir_name, 'TDErrorStates.npy'), np.asarray(outputinfo.td_states))
    np.save(os.path.join(dir_name, 'VisitStates.npy'), np.asarray(outputinfo.visit_states))
