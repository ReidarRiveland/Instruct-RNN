from instructRNN.tasks.tasks import SWAP_LIST, TASK_LIST, SUBTASKS_SWAP_DICT
from ruleLearning.rulelearner import RuleLearner, RuleLearnerConfig
from ruleLearning.rulelearner_eval import run_population_sim

import os
import itertools

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help='folder where models and data will be stored')
    parser.add_argument('exp', help='type of experiment, refering to which holdout sets to use, must be \'swap\' or \'aligned\' ')
    parser.add_argument('--models', nargs='*', help='list of model names to train, default is all models')
    parser.add_argument('--seeds', type=int, default=range(5), nargs='+', help='random seeds to use when training')
    parser.add_argument('--job_index', type=int, help='for use with slurm sbatch script, indexes the combination of seed and holdout tasks along with the model')


    args = parser.parse_args()
    MODEL_FOLDER = args.folder
    os.environ['MODEL_FOLDER']=MODEL_FOLDER
    EXP_FOLDER =MODEL_FOLDER+'/'+args.exp+'_holdouts'

    if 'go_' in args.exp: 
        task_subset_str = 'Go'
    elif 'small_' in args.exp: 
        task_subset_str = 'small'
        holdout_indices = range(len(SUBTASKS_SWAP_DICT['small']))
    else:
        task_subset_str = None
        holdout_indices = range(len(SWAP_LIST))

    num_clusters = [10, 15, 20, 25]
    jobs = list(itertools.product(args.seeds, args.models, holdout_indices, num_clusters))

    print(len(jobs))

    if args.job_index is not None: 
        jobs = [jobs[args.job_index]]

    for job in jobs: 
        seed, model_name, holdout_index, num_clusters = job 
        print(job)
    
        config = RuleLearnerConfig(
            model_name=model_name,
            holdout_index=holdout_index, 
            load_seed=seed, 
            num_clusters=num_clusters,
            task_subset_str=task_subset_str
        )
                
        run_population_sim(config)
            
                


