import os
import numpy as np
import itertools
import instructRNN.models.full_models as full_models
from instructRNN.tasks.tasks import SWAPS_DICT
from instructRNN.analysis.model_analysis import get_holdout_CCGP, get_multitask_CCGP,  get_model_clusters
from instructRNN.analysis.model_eval import get_val_perf, get_multi_all_comp_perf, get_holdout_all_comp_perf
from instructRNN.analysis.decoder_analysis import decoder_pipeline


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help='folder where models and data will be stored')
    parser.add_argument('exp', help='type of experiment, refering to which holdout sets to use, must be \'swap\' or \'aligned\' ')
    parser.add_argument('--mode', default='holdout_ccgp', help='training mode to use, must be \'train\', \'tune\', \'test\', \'decoder\' ( \'d\'),\'context\' ( \'c\')')
    parser.add_argument('--models', default=full_models.small_models, nargs='*', help='list of model names to train, default is all models')
    parser.add_argument('--seeds', type=int, default=range(5), nargs='+', help='random seeds to use when training')
    parser.add_argument('--layers', default=[layer for layer in range(1, 13)] + ['full', 'task'], nargs='+')
    parser.add_argument('--overwrite', default=False, action='store_true', help='whether or not to overwrite existing files')
    parser.add_argument('--decode_embeddings', default=False, action='store_true', help='use a decoder that directly maps embeddings to instructions')
    parser.add_argument('--job_index', type=int, help='for use with slurm sbatch script, indexes the combination of seed and holdout tasks along with the model')
    args = parser.parse_args()

    MODEL_FOLDER = args.folder
    os.environ['MODEL_FOLDER']=MODEL_FOLDER
    EXP_FOLDER =MODEL_FOLDER+'/'+args.exp+'_holdouts'

    def make_analysis_jobs(models, seeds, layers, job_index): 
        jobs = list(itertools.product(seeds, models, layers))
        if job_index is None: 
            return jobs
        else:
            return [jobs[job_index]]

    if args.mode == 'decode':
        model_name = args.models[0]
        print('processing multitask')
        decoder_pipeline(MODEL_FOLDER+'/multitask_holdouts', args.models[0], decode_embeddings=args.decode_embeddings)
        
        # if not args.decode_embeddings:
        #     print('processing sm holdouts')
        #     decoder_pipeline(MODEL_FOLDER+'/swap_holdouts', args.models[0], sm_holdout=True)
        
        #     print('processing sm holdouts and decoder holdouts')
        #     decoder_pipeline(MODEL_FOLDER+'/swap_holdouts', args.models[0], sm_holdout=True, decoder_holdout=True)

    jobs = make_analysis_jobs(args.models, args.seeds, args.layers, args.job_index)
    for job in jobs: 
        _seed, model, layer = job
        if (model in full_models.big_models or 'sbertNetL' in model) and isinstance(layer, int):
            layer += 12
        
        print(EXP_FOLDER)
        print(model)
        print(_seed)

        if args.mode == 'holdout_ccgp':
            if not os.path.exists(EXP_FOLDER+'/CCGP_scores/'+model+'/'+'layer'+str(layer)+'_task_holdout_seed'+str(_seed)+'.npy') or args.overwrite:
                print('analyzing ccgp for layer '+str(layer), flush=True)
                get_holdout_CCGP(EXP_FOLDER, model, _seed, layer= layer, save=True)
            else:
                print('Already trained: '+EXP_FOLDER+'/'+'layer'+str(layer)+'_task_holdout_seed'+str(_seed), flush=True)
                continue

        if args.mode == 'swap_ccgp':
            get_holdout_CCGP(EXP_FOLDER, model, _seed, layer= 'task', instruct_mode='swap_combined', save=True)

        elif args.mode == 'multi_ccgp': 
            get_multitask_CCGP(EXP_FOLDER, model, _seed, layer= 'task', save=True)

        elif args.mode == 'val':
            get_val_perf(EXP_FOLDER, model, _seed, save=True)

        elif args.mode == 'clusters':
            get_model_clusters(EXP_FOLDER, model, _seed, save=True)



                


