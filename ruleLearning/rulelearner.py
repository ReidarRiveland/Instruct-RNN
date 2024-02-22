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

if torch.cuda.is_available:
    device = torch.device(0)
    print(torch.cuda.get_device_name(device), flush=True)
else: 
    device = torch.device('cpu')

@define
class RuleLearnerConfig(): 
    model_name: str
    holdout_index: int
    load_seed: int 
    num_clusters: int = 15
    num_transitions: int = 2000
    switched_weight_threshold: float = 0.8
    base_alpha: float=1.0
    hard_repeat: int = 3
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
        self.model.to(device)

        print('loading inference model')
        self.inference_model = InferenceNet(len(self.in_tasks), 256)
        self.inference_model.load_state_dict(torch.load(f'inference_models/{model_path}{self.model_name}/infer_{self.model_name}_seed{self.load_seed}.pt'))
        self.inference_model.to(device)

        print('getting rule encodings')
        self.rule_encodings = self.get_known_rule_reps()
        
        print('getting direction clusters')
        self.dir_clusters, self.transition_data, self.task_transitions = self.get_task_transition_clusters()

        print('initializing memories')
        # self.task_memory = Hopfield(self.rule_encodings.mean(1), self.max_beta)
        # self.cluster_memory = Hopfield(self.dir_clusters, self.max_beta*5)

        self.combo_mat = self.get_combo_mat()

                
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
        counted_transitions = 0

        for _ in range(self.num_transitions): 
            task_draw = np.random.choice(range(len(self.in_indices)), size=2, replace=False)
            instruct_draw = np.random.choice(self.rule_encodings.shape[1], size=2)
            infer_scores = []
            for task_index in task_draw: 
                ins, targets, _, target_dirs, _ = construct_trials(self.in_tasks[task_index], 1)
                scores = self.eval_infer_net(ins)
                infer_scores.append(scores)
            
            sim_score = cosine_similarity(np.array(infer_scores))[0, 1]

            if sim_score>0.5 and self.use_weighted_transitions:
                counted_transitions += 1
                transitions.append(self.rule_encodings[task_draw[0], instruct_draw[0], : ]-self.rule_encodings[task_draw[1], instruct_draw[1], :])
                tasks.append((TASK_LIST[self.in_indices[task_draw[0]]], TASK_LIST[self.in_indices[task_draw[1]]]))
            else: 
                transitions.append(self.rule_encodings[task_draw[0], instruct_draw[0], : ]-self.rule_encodings[task_draw[1], instruct_draw[1], :])
                tasks.append((TASK_LIST[self.in_indices[task_draw[0]]], TASK_LIST[self.in_indices[task_draw[1]]]))
        
        print(counted_transitions)
        task_transition_data = np.array(transitions)
        
        return task_transition_data, tasks

    def get_task_transition_clusters(self):
        task_transition_data, tasks = self.get_task_transitions()

        k_means = KMeans(n_clusters=self.num_clusters)
        k_means.fit(task_transition_data)
        clusters = k_means.cluster_centers_

        return clusters, task_transition_data, tasks
    
    def get_combo_mat(self):
        combo_mat = np.empty((self.rule_encodings.shape[0], self.dir_clusters.shape[0], self.rule_encodings.shape[-1] ))
        for i, task_rep in enumerate(self.rule_encodings.mean(1)): 
            for j, cluster in enumerate(self.dir_clusters):
                comp_rep = task_rep+cluster
                combo_mat[i,j, :] = comp_rep

        return combo_mat

    def eval_infer_net(self, ins): 
        with torch.no_grad(): 
            inference_out, _ = self.inference_model(torch.Tensor(ins).to(self.inference_model.__device__))
        
        return softmax(inference_out[-1, -1, :].cpu().numpy())

    def eval_comp_rep(self, comp_rep, task, batch_size, trial_info=None, return_trial_inputs=False):
        if trial_info is None: 
            ins, targets, _, target_dirs, _ = construct_trials(task, batch_size)
        else: 
            ins, targets, target_dirs = trial_info

        info_embedded = torch.tensor(comp_rep).float()
        info_embedded = info_embedded[None, :].repeat(batch_size, 1)
        out, _ = self.model(torch.Tensor(ins).to(self.model.__device__), info_embedded=info_embedded)        
        rewards = isCorrect(out, torch.Tensor(targets), target_dirs)

        if return_trial_inputs: 
            return rewards, ins

        return rewards
    
    def eval_recalled_comp_reps(self, task, batch_size): 
        return np.array([np.mean(self.eval_comp_rep(rep, task, batch_size)) for rep in self.comp_rep_list]) 

    def get_all_perf_embedding_arr(self, task, batch_size=20, return_infer_out=False): 
        perf_arr = np.empty((self.combo_mat.shape[0], self.combo_mat.shape[1]))
        ins, targets, _, target_dirs, _ = construct_trials(task, batch_size)

        for i in range(self.combo_mat.shape[0]):
            for j in range(self.combo_mat.shape[1]):
                comp_rep = self.combo_mat[i, j]
                perf = np.mean(self.eval_comp_rep(comp_rep, task, batch_size, trial_info=(ins, targets, target_dirs)))
                perf_arr[i, j] = perf
        
        if return_infer_out: 
            return perf_arr, self.eval_infer_net(ins[0:5, ...])

        return perf_arr

    def update_value(self, R, alpha): 
        rpe = R-self.value
        update_factor = (alpha)*rpe
        
        self.value += (update_factor)
        return rpe, self.value

    def init_learning_variables(self, task):
        self.inhibit_mat = np.ones((self.rule_encodings.shape[0], self.dir_clusters.shape[0]))
        self.value = 0

        self.value_list = []
        self.comp_rep_list = []
        self.rpes = []
        self.task_perf = []

    def update_learning_step(self, rewards, rpe, value, comp_rep):
        self.rpes.append(rpe)
        self.value_list.append(value)
        self.comp_rep_list.append(comp_rep)
        self.task_perf += list(rewards)

    def learn(self, task, num_trials=1000): 
        self.init_learning_variables(task)

        switched_weight = 0
        task_index = np.random.randint(self.combo_mat.shape[0])
        dir_index =  np.random.randint(self.combo_mat.shape[1])
        comp_rep = self.combo_mat[np.random.randint(self.combo_mat.shape[0]), np.random.randint(self.combo_mat.shape[1])]

        for i in range(int(num_trials/self.hard_repeat)): 
            switched_weight += 1

            rewards, ins = self.eval_comp_rep(comp_rep, task, self.hard_repeat, return_trial_inputs=True)
            rpe, value = self.update_value(np.mean(rewards), (self.base_alpha/switched_weight)+1e-3) 
            self.update_learning_step(rewards, rpe, value, comp_rep)


            if (self.value<self.switched_weight_threshold):
                self.inhibit_mat[task_index, dir_index] = 0

                ###make sur eyou don't draw from task where all dir combos have been tried and are 0
                exhausted_indices = ~np.all(self.inhibit_mat==0, axis=1)
                task_probs = self.eval_infer_net(ins)[exhausted_indices]
                task_index = np.random.choice(np.arange(self.combo_mat.shape[0])[exhausted_indices], p=task_probs/task_probs.sum())


                dir_probs = self.inhibit_mat[task_index, :]/self.inhibit_mat[task_index, :].sum()
                dir_index = np.random.choice(range(self.combo_mat.shape[1]), p=dir_probs)

                comp_rep = self.combo_mat[task_index, dir_index]
                switched_weight = 0


            if i%10 == 0: 
                print('Value: ' +str(value))
            
        return (np.array(self.task_perf), 
                np.array(self.value_list), 
                np.array(self.rpes), 
                np.array(self.comp_rep_list))


