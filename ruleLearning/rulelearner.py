import torch
import numpy as np

from instructRNN.tasks.tasks import SWAP_LIST, TASK_LIST, SWAPS_DICT, construct_trials
from instructRNN.analysis.model_analysis import get_rule_reps, get_instruct_reps
from instructRNN.analysis.model_eval import task_eval, task_eval_info_embedded
from instructRNN.models.full_models import make_default_model
from instructRNN.tasks.task_criteria import isCorrect

from ruleLearning.rulelearner_utils import softmax, value_inhibit_act, circuit_recall_inhibit_act, make_periodic_beta, surround_inhibit, cos_sim

from sklearn.cluster import KMeans 
from sklearn.metrics.pairwise import cosine_similarity
from attr import define, asdict

import matplotlib.pyplot as plt


def get_held_in_indices(swap_label): 
    int_list = list(range(50))
    [int_list.remove(x) for x in [TASK_LIST.index(task) for task in SWAPS_DICT[swap_label]]]
    return int_list

@define
class RuleLearnerConfig(): 
    model_name: str
    holdout_index: int
    load_seed: int 
    num_clusters: int = 15
    num_transitions: int = 800
    max_beta: int = 25
    value_weights_init_mean: float = 0.05
    base_alpha: float = 0.25
    task_inhibit_shift: float = 0.7
    pc_inhibit_shift: float = 0.7
    switched_weight_threshold: float = 0.0
    value_inhibit_slope: float = 0.8
    hard_repeat: int = 5
    recover_act: float = 0.05

class Hopfield(): 
    def __init__(self, memory_encodings, max_beta): 
        self.max_beta = max_beta
        self.memory_encodings = memory_encodings
        self.betas = make_periodic_beta(self.max_beta)
        self.recalls = []
        self.mem_trace = []

    def set_init_pattern(self, init_pattern): 
        self.recalls = []
        self.mem_trace = [init_pattern]

    def probe_memory(self, probe_pattern, value_inhibit_w, beta): 
        dotted = np.matmul(probe_pattern, self.memory_encodings.T)
        inhibited_dotted = np.multiply(dotted, value_inhibit_w)

        softmaxed = softmax(inhibited_dotted, beta=beta)
        recalled = np.matmul(softmaxed, self.memory_encodings)
        return recalled, softmaxed

    def recall_memory(self, value_inhibit_w):
        assert len(self.mem_trace)>0, 'MUST SET THE INIT MEM TRACE REP WITH set_init_pattern'
        probe_pattern = self.mem_trace[-1]

        #for i, beta in enumerate(self.betas): 
        probe_pattern, softmaxed = self.probe_memory(probe_pattern+np.random.normal(scale=0.01, size=self.memory_encodings.shape[-1]), value_inhibit_w, self.max_beta) 
        self.mem_trace.append(probe_pattern)

        recalled_pattern = probe_pattern
        post_activity = softmaxed
        self.recalls.append(recalled_pattern)

        return recalled_pattern, post_activity

class RuleLearner(): 
    def __init__(self, config): 
        self.config = asdict(config, recurse=False)
        for name, value in self.config.items(): 
            setattr(self, name, value)

        self.in_tasks = [TASK_LIST[i] for i in get_held_in_indices(f'swap{self.holdout_index}')]

        print('loading model')
        self.model = make_default_model(self.model_name)
        self.model.load_model(f'NN_simData/swap_holdouts/swap{self.holdout_index}/{self.model_name}', suffix=f'_seed{self.load_seed}')

        print('getting rule encodings')
        self.rule_encodings = self.get_holdout_rule_reps()
        
        print('getting direction clusters')
        self.dir_clusters, self.transition_data, self.task_transitions = self.get_task_transition_clusters()

        print('initializing memories')
        self.task_memory = Hopfield(self.rule_encodings.mean(1), self.max_beta)
        self.cluster_memory = Hopfield(self.dir_clusters, self.max_beta*5)

                
    def get_holdout_rule_reps(self, depth='full'): 
        held_in_indices = get_held_in_indices(f'swap{self.holdout_index}')    

        if hasattr(self.model, 'langModel'): 
            rule_reps = get_instruct_reps(self.model.langModel, depth=depth)
        elif hasattr(self.model, 'rule_encoder'): 
            rule_reps = get_rule_reps(self.model, rule_layer=depth)
        else: 
            rule_reps = get_rule_reps(self.model)
            
        return rule_reps[held_in_indices, ...]

    def get_task_transition_clusters(self):
        transitions = []
        tasks = []
        in_indices = get_held_in_indices(f'swap{self.holdout_index}')

        for _ in range(self.num_transitions): 
            task_draw = np.random.choice(range(len(in_indices)), size=2, replace=False)
            instruct_draw = np.random.choice(self.rule_encodings.shape[1], size=2)

            transitions.append(self.rule_encodings[task_draw[0], instruct_draw[0], : ]-self.rule_encodings[task_draw[1], instruct_draw[1], :])
            tasks.append((TASK_LIST[in_indices[task_draw[0]]], TASK_LIST[in_indices[task_draw[1]]]))
        
        task_transition_data = np.array(transitions)

        k_means = KMeans(n_clusters=self.num_clusters)
        k_means.fit(task_transition_data)
        clusters = k_means.cluster_centers_

        return clusters, task_transition_data, tasks

    def visual_inhibition_curve(self, curve): 
        x = np.linspace(0, 1)

        if curve == 'circuit': 
            y = [circuit_recall_inhibit_act(_x, threshold=self.circuit_inhibit_threshold) for _x in x]
        elif curve == 'value': 
            y = value_inhibit_act(x, shift=self.task_inhibit_shift, slope=self.value_inhibit_slope)

        plt.plot(x,y)
        plt.show()

    def get_base_rule_encodings(self):
        rule_encoding_set = np.delete(self.rule_encodings, 
                    np.array([TASK_LIST.index(holdout_task) for holdout_task in SWAP_LIST[self.holdout_index]]), 
                    axis=0)

        return rule_encoding_set

    def get_base_rule_PCs(self): 
        embedder = PCA()
        rule_encoding_set = self.get_base_rule_encodings()
        embedder.fit(rule_encoding_set.reshape(-1, rule_encoding_set.shape[-1]))
        componenets = embedder.components_[:self.num_clusters, :]
        return componenets


    def eval_comp_rep(self, comp_rep, task, batch_size):
        # if hasattr(self.model, 'langModel'):
        #     #info_embedded = self.model.langModel.proj_out(torch.tensor(comp_rep).float())
        #     info_embedded = torch.tensor(comp_rep).float()
        # else:
        #     info_embedded = self.model.rule_encoder.rule_layer2(torch.tensor(comp_rep).float())


        info_embedded = torch.tensor(comp_rep).float()
        info_embedded = info_embedded[None, :].repeat(batch_size, 1)
        ins, targets, _, target_dirs, _ = construct_trials(task, batch_size)
        out, _ = self.model(torch.Tensor(ins).to(self.model.__device__), info_embedded=info_embedded)        
        rewards = isCorrect(out, torch.Tensor(targets), target_dirs)
        return rewards
    
    def eval_recalled_comp_reps(self, task, batch_size): 
        return np.array([np.mean(self.eval_comp_rep(rep, task, 50)) for rep in self.comp_rep_list]) 

    def get_all_perf_embedding_arr(self, task, batch_size=50): 
        perf_arr = np.empty((self.cluster_memory.memory_encodings.shape[0], self.task_memory.memory_encodings.shape[0]))

        for i, pc in enumerate(self.cluster_memory.memory_encodings):
            print('processing pc ' + str(i))
            for j, task_rep in enumerate(self.task_memory.memory_encodings): 
                comp_rep = task_rep+pc
                perf = np.mean(self.eval_comp_rep(comp_rep, task, batch_size))
                perf_arr[i, j] = perf
        
        return perf_arr

    def update_inhibition(self, value_weight, post_activity, inhibit_weights, recover_act=0.01): 
        post_activity = surround_inhibit(post_activity, recover_act)
        post_activity = value_weight*post_activity
        inhibit_weights = np.minimum(1, np.maximum(0, inhibit_weights-post_activity))
        return inhibit_weights

    def update_value_weights(self, embedding, R, alpha): 
        rpe = R-np.dot(self.W_v.T, embedding)
        update_factor = (alpha)*rpe
        
        self.W_v += (update_factor*embedding)
        return rpe

    def update_value(self, R, alpha): 
        rpe = R-self.value
        update_factor = (alpha)*rpe
        
        self.value += (update_factor)
        return rpe

    def init_learning_variables(self, task):
        #self.W_v = np.random.normal(scale=0.01, size=self.rule_encodings.shape[-1])
        self.W_v = np.zeros(self.rule_encodings.shape[-1])
        self.value = 0

        self.task_inhibit_history = []
        self.pc_inhibit_history = []

        self.task_value_inhibit_w = np.ones(len(self.in_tasks))
        self.pc_value_inhibit_w = np.ones(self.num_clusters)

        self.task_memory.set_init_pattern(self.task_memory.memory_encodings[np.random.randint(0, 45), :])
        self.cluster_memory.set_init_pattern(self.cluster_memory.memory_encodings[0, :])

        self.value_list = []
        self.comp_rep_list = []
        self.rpes = []
        self.task_perf = []

    def update_learning_step(self, rewards, rpe, value, comp_rep):
        self.rpes.append(rpe)
        self.value_list.append(value)
        self.comp_rep_list.append(comp_rep)
        self.task_perf += list(rewards)
        self.task_inhibit_history.append(self.task_value_inhibit_w)
        self.pc_inhibit_history.append(self.pc_value_inhibit_w)

    def learn(self, task, num_trials=1000): 
        self.init_learning_variables(task)
        switched_weight = 0
        comp_rep = self.task_memory.mem_trace[0]
        task_post_activity = np.zeros(45)
        pc_post_activity = np.zeros(self.num_clusters)

        for i in range(int(num_trials/self.hard_repeat)): 
            switched_weight += 1

            rewards = self.eval_comp_rep(comp_rep, task, self.hard_repeat)
            #rpe = self.update_value_weights(comp_rep, np.mean(rewards), (self.base_alpha/switched_weight)+1e-3) 
            rpe = self.update_value(np.mean(rewards), (self.base_alpha/switched_weight)+1e-3) 
            #value = np.dot(self.W_v.T, comp_rep)
            value = self.value 
            self.update_learning_step(rewards, rpe, value, comp_rep)

            task_inhbition_weight = value_inhibit_act(value, shift=self.task_inhibit_shift, slope=self.value_inhibit_slope)
            pc_inhbition_weight = value_inhibit_act(value, shift=self.pc_inhibit_shift, slope=self.value_inhibit_slope)

            self.task_value_inhibit_w = self.update_inhibition(task_inhbition_weight, task_post_activity, self.task_value_inhibit_w, recover_act=self.recover_act)
            self.pc_value_inhibit_w = self.update_inhibition(pc_inhbition_weight, pc_post_activity, self.pc_value_inhibit_w, recover_act=self.recover_act)
            task_recalled, task_post_activity = self.task_memory.recall_memory(self.task_value_inhibit_w)
            pc_recalled, pc_post_activity = self.cluster_memory.recall_memory(self.pc_value_inhibit_w)

            comp_rep = task_recalled+pc_recalled

            if (task_inhbition_weight>self.switched_weight_threshold) and (pc_inhbition_weight>self.switched_weight_threshold):
                switched_weight = 0 

            if i%10 == 0: 
                print('Value: ' +str(value))
            
        return (np.array(self.task_perf), 
                np.array(self.value_list), 
                np.array(self.rpes), 
                np.array(self.comp_rep_list), 
                np.array(self.task_inhibit_history),
                np.array(self.pc_inhibit_history))

