import os
import numpy as np
import itertools
import instructRNN.models.full_models as full_models
from instructRNN.analysis.model_analysis import get_holdout_CCGP, get_multitask_CCGP, get_layer_dim, get_val_perf, get_model_clusters
from instructRNN.analysis.decoder_analysis import decoder_pipeline

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help='folder where models and data will be stored')
    parser.add_argument('exp', help='type of experiment, refering to which holdout sets to use, must be \'swap\' or \'aligned\' ')
    parser.add_argument('--mode', default='holdout_ccgp', help='training mode to use, must be \'train\', \'tune\', \'test\', \'decoder\' ( \'d\'),\'context\' ( \'c\')')
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

    if args.mode == 'decode':
        decoder_pipeline('7.20models/multitask_holdouts', 'clipNet_lin')
        decoder_pipeline('7.20models/swap_holdouts', 'clipNet_lin', sm_holdout=True)
        decoder_pipeline('7.20models/swap_holdouts', 'clipNet_lin', sm_holdout=True, decoder_holdout=True)



    jobs = make_analysis_jobs(args.models, args.seeds, args.job_index)
    for job in jobs: 
        _seed, model = job
        if model in full_models.shallow_models: 
            layer_list = ['task']
        elif model in full_models.big_models: 
            layer_list = [str(layer) for layer in range(12, 25)] + ['full', 'task']
        elif 'bow' in model: 
            layer_list = ['bow', 'full', 'task']
        else: 
            layer_list = [str(layer) for layer in range(1, 13)] + ['full', 'task']
        
        print(EXP_FOLDER)
        print(model)
        print(_seed)


        if args.mode == 'holdout_ccgp':
            for layer in layer_list[::-1]:
                try:
                    np.load(EXP_FOLDER+'/CCGP_scores/'+model+'/'+'layer'+layer+'_task_holdout_seed'+str(_seed)+'.npy')
                    print('Already trained: '+EXP_FOLDER+'/'+'layer'+layer+'_task_holdout_seed'+str(_seed))
                    continue
                except FileNotFoundError:
                    get_holdout_CCGP(EXP_FOLDER, model, _seed, layer= layer, save=True)

        if args.mode == 'swap_ccgp':
            get_holdout_CCGP(EXP_FOLDER, model, _seed, layer= 'task', instruct_mode='swap', save=True)

        elif args.mode == 'multi_ccgp': 
            get_multitask_CCGP(EXP_FOLDER, model, _seed, layer= 'task', save=True)

        elif 'dim' in args.mode:    
            for layer in layer_list:
                get_layer_dim(EXP_FOLDER, model, _seed, layer= layer, save=True)
                
        elif args.mode == 'val':
            get_val_perf(EXP_FOLDER, model, _seed, save=True)
                
        elif args.mode == 'clusters':
            get_model_clusters(EXP_FOLDER, model, _seed, save=True)
                


