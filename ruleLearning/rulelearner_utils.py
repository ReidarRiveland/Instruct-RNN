import numpy as np
import matplotlib.pyplot as plt   
from ruleLearning.rulelearner_plotting import * 

def make_periodic_beta(max_beta):
    return max_beta/2*np.cos(np.linspace(-np.pi, np.pi, 11, endpoint=False))+max_beta/2

def softmax(x, beta=1):
    return np.exp(beta*x)/np.sum(np.exp(beta*x))

def circuit_recall_inhibit_act(x, threshold=0.7): 
    if x>=threshold: return 1
    else: return np.maximum(0,x)

def value_inhibit_act(x, shift=0.5, slope=8): 
    return -1*np.tanh((x-shift)*slope)

def surround_inhibit(post_activity, recover_act): 
    surround_inhibit = -1*np.full_like(post_activity, recover_act)
    surround_inhibit[post_activity.argmax()] = post_activity.max()
    return surround_inhibit

def cos_sim(x, y): 
    return np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))

class RuleLearningData(): 
    DATA_KEYS = ['rewards', 'values', 'rpes', 'comp_reps']
    def __init__(self, model, seed, num_clusters): 
        self.model = model
        self.seed = seed 
        self.num_clusters = num_clusters
        self.task_list = SUBTASKS_DICT['small'][:-4]
        self.rewards = np.empty((len(self.task_list), 15, 801))
        self.values = np.empty((len(self.task_list), 15, 267))
        self.rpes = np.empty((len(self.task_list), 15, 267))
        self.comp_reps = np.empty((len(self.task_list), 15, 267, 64))

        for i, task in enumerate(self.task_list): 
            rulelearning_data = np.load(f'rulelearner_logs/{self.model}/seed{self.seed}_{self.num_clusters}/{task}_rulelearning.npz')
            for key in self.DATA_KEYS: 
                data = getattr(self, key)
                data[i, ...] = rulelearning_data[key]

    def get_asym_perf(self, num_steps=100): 
        return self.rewards[:, :, -num_steps:]

    def plot_all_learning_curves(self, task): 
        plot_population_learning(self.rewards[self.task_list.index(task)], task)
        plt.show()

    def plot_task_learning(self, task, pop_index): 
        task_index = self.task_list.index(task)
        plot_task_learning(task, self.rewards[task_index, pop_index, :], 
                                self.values[task_index, pop_index, :], 
                                self.rpes[task_index, pop_index, :], 
                                self.comp_reps[task_index, pop_index, :], 
                                hard_repeat=3, embedding_perfs=None)