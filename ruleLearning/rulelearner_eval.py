from ruleLearner.models.full_models import make_default_model
from ruleLearner.tasks.tasks import SWAP_LIST, TASK_LIST, construct_trials, SWAPS_DICT
from ruleLearner.rule_learner.rulelearner import RuleLearner, RuleLearnerConfig
from ruleLearner.rule_learner.rulelearner_plotting import plot_task_learning, plot_memory_trace, plot_population_learning

import numpy as np 
from datetime import datetime
from attr import asdict
import json
import os

def init_logging_dir(sim_label, config):
    now = datetime.now()
    dt_str = now.strftime("%d.%m.%Y.%H.%M")
    log_dir = 'rulelearner_logs/' + sim_label +'/' + dt_str
    if not os.path.isdir(log_dir): 
        os.makedirs(log_dir)
    with open(log_dir+'/config_dict', 'w', encoding='utf8') as fp:
        json.dump(asdict(config), fp)
    return log_dir

def get_population_learning(rulelearner, task, pop_size=10, num_trials=1200): 
    assert np.remainder(num_trials, rulelearner.hard_repeat) == 0, 'number of trials must be divisible by rule learner hard repeat'
    num_batches = int(num_trials/rulelearner.hard_repeat)

    pop_reward = np.empty((pop_size, num_trials))
    pop_values = np.empty((pop_size, num_batches))
    pop_rpes = np.empty((pop_size, num_batches))
    pop_comp_reps = np.empty((pop_size, num_batches, rulelearner.all_rule_encodings.shape[-1]))
    pop_task_inhibit = np.empty((pop_size, num_batches, rulelearner.task_memory.memory_encodings.shape[0]))
    pop_pc_inhibit = np.empty((pop_size, num_batches, rulelearner.num_pcs))

    for i in range(pop_size): 
        print('Evaluating Pop. Member '+str(i))
        #task_rewards, values, rpes, comp_reps, task_inhibit_history, pc_inhibit_history = rulelearner.learn(task, num_trials=num_trials)
        results = rulelearner.learn(task, num_trials=num_trials)
        pop_reward[i, ...] = results[0]
        pop_values[i, ...] = results[1]
        pop_rpes[i, ...] = results[2]
        pop_comp_reps[i, ...] = results[3]
        pop_task_inhibit[i, ...] = results[4]
        pop_pc_inhibit[i, ...] = results[5]

    return pop_reward, pop_values, pop_rpes, pop_comp_reps, pop_task_inhibit, pop_pc_inhibit
    
def run_population_sim(foldername, model_name, rulelearner_config, seed=0): 
    model = make_default_model(model_name)
    log_dir = init_logging_dir(model.model_name+'/seed'+str(0)+'/', rulelearner_config)

    for holdout_index, holdout_task in enumerate(SWAPS_DICT.values()):
        model.load_model(foldername+'/swap'+str(holdout_index)+'/'+model.model_name, suffix='_seed'+str(seed))
        rulelearner = RuleLearner(model, holdout_index, rulelearner_config)
        for task in holdout_task:
            print('Evaluating Task ' + task)
            pop_results = get_population_learning(rulelearner, task)

            np.savez(log_dir+'/'+task+'_rulelearning.npz', rewards = pop_results[0],
                                            values = pop_results[1],
                                            rpes = pop_results[2],
                                            comp_reps = pop_results[3],
                                            task_inhibit = pop_results[4], 
                                            pc_inhibit = pop_results[5])

