import itertools

from torch import seed
from instructRNN.trainers.model_trainer import *
from instructRNN.trainers.decoder_trainer import *
from instructRNN.trainers.context_trainer import *

from instructRNN.tasks.tasks import SWAPS_DICT, ALIGNED_DICT
from instructRNN.models.full_models import all_models, untuned_models, tuned_models

def make_training_jobs(exp, models, seeds, holdouts, job_index):
    if exp == 'swap': 
        _holdout_dict = SWAPS_DICT
    elif args.exp == 'algined': 
        _holdout_dict = ALIGNED_DICT

    if holdouts is None: 
        holdout_dict = _holdout_dict
    else: 
        holdout_dict = dict([list(_holdout_dict.items())[i] for i in args.holdouts])

    jobs = list(itertools.product(models, seeds, holdout_dict.items()))
    if job_index is None: 
        return jobs
    else:
        return [jobs[job_index]]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help='folder where models and data will be stored')
    parser.add_argument('exp', help='type of experiment, refering to which holdout sets to use, must be \'swap\' or \'aligned\' ')
    parser.add_argument('--mode', default='train', help='training mode to use, must be \'train\', \'tune\', \'test\', \'decoder\' ( \'d\'),\'context\' ( \'c\')')
    parser.add_argument('--models', nargs='*', help='list of model names to train, default is all models')
    parser.add_argument('--holdouts', type=int, default=None,  nargs='*', help='list of ints that index the holdout sets to use')
    parser.add_argument('--overwrite', default=False, action='store_true', help='whether or not to overwrite existing files')
    parser.add_argument('--seeds', type=int, default=[0], nargs='+', help='random seeds to use when training')
    parser.add_argument('--layer', default='last', help='the dim corresponding to the layer the contexts gets trained at, \
                                                    must be emd or last, only for use if mode is context')
    parser.add_argument('--use_holdouts', default=False, action='store_true', help='whether to holdout tasks instructions in training decoders')
    parser.add_argument('--job_index', type=int, help='for use with slurm sbatch script, indexes the combination of seed and holdout tasks along with the model')
    args = parser.parse_args()

    MODEL_FOLDER = args.folder
    EXP_FOLDER =MODEL_FOLDER+'/'+args.exp+'_holdouts'

    jobs = make_training_jobs(args.exp, args.models, args.seeds, args.holdouts, args.job_index)
    
    for job in jobs: 
        model, _seed, holdouts = job
        if args.mode == 'train': 
            train_model(EXP_FOLDER, model, _seed, holdouts, overwrite=args.overwrite)     
        if args.mode == 'tune': 
            tune_model(EXP_FOLDER, model, _seed, holdouts, overwrite=args.overwrite)     
        if args.mode == 'test': 
            test_model(EXP_FOLDER, model, _seed, holdouts, overwrite=args.overwrite)   
        if args.mode == 'context' or args.mode == 'c': 
            train_contexts(EXP_FOLDER, model, _seed, holdouts, args.layer, overwrite=args.overwrite, 
                                lr=0.005, num_contexts=5, tasks=TASK_LIST[::-1][7:])
        if args.mode == 'decoder' or args.mode == 'd': 
            train_decoder(EXP_FOLDER, model, _seed, holdouts, args.use_holdouts, overwrite=args.overwrite)