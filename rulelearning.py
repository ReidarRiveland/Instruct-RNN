from ruleLearner.models.full_models import SimpleNetPlus
from ruleLearner.tasks.tasks import SWAP_LIST, TASK_LIST, construct_trials
from ruleLearner.rule_learner.rulelearner import RuleLearner, RuleLearnerConfig
from ruleLearner.rule_learner.rulelearner_eval import run_population_sim

import os

def make_training_jobs(exp, models, seeds, holdouts, job_index):
    if exp == 'swap': 
        _holdout_dict = SWAPS_DICT
    elif args.exp == 'aligned': 
        _holdout_dict = ALIGNED_DICT
    elif args.exp == 'family': 
        _holdout_dict = FAMILY_DICT
    elif args.exp == 'multitask': 
        _holdout_dict = MULTITASK_DICT

    if holdouts is None: 
        holdout_dict = _holdout_dict
    else: 
        holdout_dict = dict([list(_holdout_dict.items())[i] for i in args.holdouts])

    jobs = list(itertools.product(seeds, models, holdout_dict.items()))

    if job_index is None: 
        return jobs
    else:
        return [jobs[job_index]]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help='folder where models and data will be stored')
    parser.add_argument('exp', help='type of experiment, refering to which holdout sets to use, must be \'swap\' or \'aligned\' ')
    parser.add_argument('--mode', default='param_grid', help='training mode to use, must be \'train\', \'tune\', \'test\', \'decoder\' ( \'d\'),\'context\' ( \'c\')')
    parser.add_argument('--model', default='simpleNetPlus')
    parser.add_argument('--holdouts', type=int, default=None,  nargs='*', help='list of ints that index the holdout sets to use')
    parser.add_argument('--seeds', type=int, default=range(5), nargs='+', help='random seeds to use when training')

    args = parser.parse_args()


    MODEL_FOLDER = args.folder
    os.environ['MODEL_FOLDER']=MODEL_FOLDER
    EXP_FOLDER =MODEL_FOLDER+'/'+args.exp+'_holdouts'

    if args.mode == 'param_grid': 

        if args.model == 'simpleNetPlus':
            beta_range = [5, 10, 25, 50, 80, 100, 150]
            slope_range = [0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 50.0]

            for beta in beta_range: 
                for slope in slope_range: 
                    config = RuleLearnerConfig(
                        value_inhibit_slope = slope,
                        max_beta=beta,
                    )
                    print('here')
                    run_population_sim(EXP_FOLDER, args.model, config)
        
                
        elif args.model == 'clipNet_lin':
            beta_range = [1, 3, 5, 10, 15, 20, 25]
            slope_range = [0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 40.0, 50.0]

            for beta in beta_range: 
                for slope in slope_range: 
                    config = RuleLearnerConfig(
                        base_alpha=0.05,
                        value_weights_init_mean=0.01,
                        value_inhibit_slope = slope,
                        max_beta=beta,
                    )
                    run_population_sim(EXP_FOLDER, args.model, config)
        
                


