from instructRNN.models.full_models import make_default_model
from instructRNN.tasks.tasks import SWAP_LIST, TASK_LIST, construct_trials, SWAPS_DICT
from ruleLearning.rulelearner import RuleLearner, RuleLearnerConfig
from ruleLearning.rulelearner_plotting import plot_task_learning, plot_memory_trace, plot_population_learning

import numpy as np 
from datetime import datetime
from attr import asdict
import json
import os

def init_logging_dir(config):
    log_dir = f'rulelearner_logs/{config.model_name}/seed{config.load_seed}_{config.num_clusters}/'
    if not os.path.isdir(log_dir): 
        os.makedirs(log_dir)
    with open(log_dir+'/config_dict', 'w', encoding='utf8') as fp:
        json.dump(asdict(config), fp)
    return log_dir

def get_population_learning(rulelearner, task, pop_size=15, num_trials=801): 
    assert np.remainder(num_trials, rulelearner.hard_repeat) == 0, 'number of trials must be divisible by rule learner hard repeat'
    num_batches = int(num_trials/rulelearner.hard_repeat)

    pop_reward = np.empty((pop_size, num_trials))
    pop_values = np.empty((pop_size, num_batches))
    pop_rpes = np.empty((pop_size, num_batches))
    pop_comp_reps = np.empty((pop_size, num_batches, rulelearner.combo_mat.shape[-1]))


    for i in range(pop_size): 
        print('Evaluating Pop. Member '+str(i))
        #task_rewards, values, rpes, comp_reps, task_inhibit_history, pc_inhibit_history = rulelearner.learn(task, num_trials=num_trials)
        results = rulelearner.learn(task, num_trials=num_trials)
        pop_reward[i, ...] = results[0]
        pop_values[i, ...] = results[1]
        pop_rpes[i, ...] = results[2]
        pop_comp_reps[i, ...] = results[3]

    return pop_reward, pop_values, pop_rpes, pop_comp_reps
    
def run_population_sim(rulelearner_config): 
    log_dir = init_logging_dir(rulelearner_config)

    rulelearner = RuleLearner(rulelearner_config)
    for task in rulelearner.holdouts:
        print('Evaluating Task ' + task)
        pop_results = get_population_learning(rulelearner, task)

        np.savez(log_dir+task+'_rulelearning.npz', rewards = pop_results[0],
                                        values = pop_results[1],
                                        rpes = pop_results[2],
                                        comp_reps = pop_results[3])

