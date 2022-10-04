import os
import itertools
import instructRNN.models.full_models as full_models
from instructRNN.analysis.model_analysis import get_holdout_CCGP

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help='folder where models and data will be stored')
    parser.add_argument('exp', help='type of experiment, refering to which holdout sets to use, must be \'swap\' or \'aligned\' ')
    parser.add_argument('--mode', default='ccgp', help='training mode to use, must be \'train\', \'tune\', \'test\', \'decoder\' ( \'d\'),\'context\' ( \'c\')')
    parser.add_argument('--models', default=full_models.small_models, nargs='*', help='list of model names to train, default is all models')
    parser.add_argument('--seeds', type=int, default=range(5), nargs='+', help='random seeds to use when training')

    parser.add_argument('--job_index', type=int, help='for use with slurm sbatch script, indexes the combination of seed and holdout tasks along with the model')
    args = parser.parse_args()

    MODEL_FOLDER = args.folder
    os.environ['MODEL_FOLDER']=MODEL_FOLDER
    EXP_FOLDER =MODEL_FOLDER+'/'+args.exp+'_holdouts'

    def make_analysis_jobs(models, seeds, job_index): 
        jobs = list(itertools.product(seeds, models))
        if job_index is None: 
            return jobs
        else:
            return [jobs[job_index]]

    jobs = make_analysis_jobs(args.models, args.seeds, args.job_index)
    print(jobs)
    for job in jobs: 
        _seed, model = job
        if args.mode == 'ccgp': 
            from instructRNN.analysis.model_analysis import * 

            if model in full_models.shallow_models: 
                layer_list = ['task']
            elif model in full_models.big_models: 
                layer_list = [str(layer) for layer in range(1, 25)] + ['full', 'task']
            elif 'bow' in model: 
                layer_list = ['bow', 'full', 'task']
            else: 
                layer_list = [str(layer) for layer in range(1, 13)] + ['full', 'task']
            
            print(EXP_FOLDER)
            print(model)
            print(_seed)
            print(layer_list)
            for layer in layer_list:
                get_holdout_CCGP(EXP_FOLDER, model, _seed, layer= layer, save=True)

