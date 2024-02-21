import torch
import numpy as np

from instructRNN.tasks.tasks import SWAP_LIST, TASK_LIST, SWAPS_DICT, construct_trials, SUBTASKS_DICT, SUBTASKS_SWAP_DICT
from instructRNN.analysis.model_analysis import get_rule_reps, get_instruct_reps
from instructRNN.analysis.model_eval import task_eval, task_eval_info_embedded
from instructRNN.models.full_models import make_default_model
from instructRNN.tasks.task_criteria import isCorrect
from instructRNN.models.sensorimotor_models import InferenceNet

from ruleLearning.rulelearner_utils import softmax, value_inhibit_act, circuit_recall_inhibit_act, make_periodic_beta, surround_inhibit, cos_sim

from sklearn.cluster import KMeans 
from sklearn.metrics.pairwise import cosine_similarity
from attr import define, asdict

import matplotlib.pyplot as plt

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
    task_subset_str: str = None
    use_weighted_transitions: bool = True

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
        probe_pattern, softmaxed = self.probe_memory(probe_pattern+np.random.normal(scale=0.01, 
                                                        size=self.memory_encodings.shape[-1]), 
                                                        value_inhibit_w, self.max_beta) 
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

        if self.task_subset_str is None: 
            self.task_list = TASK_LIST
            label_str = f"swap{self.holdout_index}"
            self.holdouts = SWAPS_DICT[label_str]
            self.in_tasks = [task for task in TASK_LIST if task not in self.holdouts]
            model_path = f'NN_simData/swap_holdouts/{label_str}/'
        else: 
            label_str = f"{self.task_subset_str}_swap{self.holdout_index}"
            self.task_list = SUBTASKS_DICT[self.task_subset_str]
            self.holdouts = SUBTASKS_SWAP_DICT[self.task_subset_str][label_str]
            model_path = f'SUB_SIM/{self.task_subset_str}_swap_holdouts/{label_str}/'

        self.in_tasks = [task for task in self.task_list if task not in self.holdouts]
        self.in_indices = [self.task_list.index(task) for task in self.in_tasks]

        print('loading model')
        self.model = make_default_model(self.model_name)
        self.model.load_model(model_path + self.model_name, suffix=f'_seed{self.load_seed}')

        print('loading inference model')
        self.inference_model = InferenceNet(len(self.in_tasks), 256)
        self.inference_model.load_state_dict(torch.load(f'inference_models/{model_path}{self.model_name}/infer_{self.model_name}_seed{self.load_seed}.pt'))

        print('getting rule encodings')
        self.rule_encodings = self.get_known_rule_reps()
        
        print('getting direction clusters')
        self.dir_clusters, self.transition_data, self.task_transitions = self.get_task_transition_clusters()

        print('initializing memories')
        self.task_memory = Hopfield(self.rule_encodings.mean(1), self.max_beta)
        self.cluster_memory = Hopfield(self.dir_clusters, self.max_beta*5)

                
    def get_known_rule_reps(self, depth='full'): 
        if hasattr(self.model, 'langModel'): 
            rule_reps = get_instruct_reps(self.model.langModel, depth=depth)
        elif hasattr(self.model, 'rule_encoder'): 
            rule_reps = get_rule_reps(self.model, rule_layer=depth)
        else: 
            rule_reps = get_rule_reps(self.model)
            
        return rule_reps[self.in_indices, ...]

    def get_task_transitions(self): 
        transitions = []
        tasks = []

        for _ in range(self.num_transitions): 
            task_draw = np.random.choice(range(len(self.in_indices)), size=2, replace=False)
            instruct_draw = np.random.choice(self.rule_encodings.shape[1], size=2)

            transitions.append(self.rule_encodings[task_draw[0], instruct_draw[0], : ]-self.rule_encodings[task_draw[1], instruct_draw[1], :])
            tasks.append((TASK_LIST[self.in_indices[task_draw[0]]], TASK_LIST[self.in_indices[task_draw[1]]]))
        
        task_transition_data = np.array(transitions)

        return task_transition_data, tasks

    def get_weighted_task_transitions(self): 
        transitions = []
        tasks = []
        counted_transitions = 0

        for _ in range(self.num_transitions): 
            task_draw = np.random.choice(range(len(self.in_indices)), size=2, replace=False)
            instruct_draw = np.random.choice(self.rule_encodings.shape[1], size=2)
            infer_scores = []
            for task_index in task_draw: 
                ins, targets, _, target_dirs, _ = construct_trials(self.in_tasks[task_index], 1)
                with torch.no_grad(): infer_out, _ = self.inference_model(torch.Tensor(ins).to(self.inference_model.__device__))
                infer_scores.append(softmax(infer_out[0, -1, :].numpy()))
            
            
            sim_score = cosine_similarity(np.array(infer_scores))[0, 1]

            if sim_score>0.5:
                counted_transitions += 1
                transitions.append(self.rule_encodings[task_draw[0], instruct_draw[0], : ]-self.rule_encodings[task_draw[1], instruct_draw[1], :])
                tasks.append((TASK_LIST[self.in_indices[task_draw[0]]], TASK_LIST[self.in_indices[task_draw[1]]]))
            
        task_transition_data = np.array(transitions)
        print(counted_transitions)

        return task_transition_data, tasks

    def get_task_transition_clusters(self):
        if self.use_weighted_transitions: 
            task_transition_data, tasks = self.get_weighted_task_transitions()
        else: 
            task_transition_data, tasks = self.get_task_transitions()

        k_means = KMeans(n_clusters=self.num_clusters)
        k_means.fit(task_transition_data)
        clusters = k_means.cluster_centers_

        return clusters, task_transition_data, tasks

    def eval_comp_rep(self, comp_rep, task, batch_size, trial_info=None):
        if trial_info is None: 
            ins, targets, _, target_dirs, _ = construct_trials(task, batch_size)
        else: 
            ins, targets, target_dirs = trial_info

        info_embedded = torch.tensor(comp_rep).float()
        info_embedded = info_embedded[None, :].repeat(batch_size, 1)
        out, _ = self.model(torch.Tensor(ins).to(self.model.__device__), info_embedded=info_embedded)        
        rewards = isCorrect(out, torch.Tensor(targets), target_dirs)
        return rewards
    
    def eval_recalled_comp_reps(self, task, batch_size): 
        return np.array([np.mean(self.eval_comp_rep(rep, task, batch_size)) for rep in self.comp_rep_list]) 

    def get_all_perf_embedding_arr(self, task, batch_size=20, return_infer_out=False): 
        perf_arr = np.empty((self.cluster_memory.memory_encodings.shape[0], self.task_memory.memory_encodings.shape[0]))
        ins, targets, _, target_dirs, _ = construct_trials(task, batch_size)

        for i, cluster in enumerate(self.cluster_memory.memory_encodings):
            print('processing cluster ' + str(i))
            for j, task_rep in enumerate(self.task_memory.memory_encodings): 
                comp_rep = task_rep+cluster
                perf = np.mean(self.eval_comp_rep(comp_rep, task, batch_size, trial_info=(ins, targets, target_dirs)))
                perf_arr[i, j] = perf
        
        if return_infer_out: 
            with torch.no_grad(): inference_out, _ = self.inference_model(torch.Tensor(ins).to(self.inference_model.__device__))
            return perf_arr, inference_out[:, -1, :].numpy()

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

        self.task_memory.set_init_pattern(self.task_memory.memory_encodings[np.random.randint(0, len(self.in_indices)), :])
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
        task_post_activity = np.zeros(len(self.in_indices))
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


    def visual_inhibition_curve(self, curve): 
        x = np.linspace(0, 1)

        if curve == 'circuit': 
            y = [circuit_recall_inhibit_act(_x, threshold=self.circuit_inhibit_threshold) for _x in x]
        elif curve == 'value': 
            y = value_inhibit_act(x, shift=self.task_inhibit_shift, slope=self.value_inhibit_slope)

        plt.plot(x,y)
        plt.show()
