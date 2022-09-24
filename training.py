import os
import itertools
from instructRNN.tasks.tasks import MULTITASK_DICT, SWAPS_DICT, ALIGNED_DICT, FAMILY_DICT
from instructRNN.models.full_models import small_models

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
    parser.add_argument('--mode', default='pipeline', help='training mode to use, must be \'train\', \'tune\', \'test\', \'decoder\' ( \'d\'),\'context\' ( \'c\')')
    parser.add_argument('--models', default=small_models, nargs='*', help='list of model names to train, default is all models')
    parser.add_argument('--holdouts', type=int, default=None,  nargs='*', help='list of ints that index the holdout sets to use')
    parser.add_argument('--overwrite', default=False, action='store_true', help='whether or not to overwrite existing files')
    parser.add_argument('--use_checkpoint', default=False, action='store_true', help='whether or not to use checkpointed model')

    parser.add_argument('--ot', default=False, action='store_true', help='retest')
    parser.add_argument('--seeds', type=int, default=range(5), nargs='+', help='random seeds to use when training')
    
    parser.add_argument('--layer', default='emb', help='the dim corresponding to the layer the contexts gets trained at, \
                                                    must be emd or last, only for use if mode is context')
    parser.add_argument('--use_holdouts', default=False, action='store_true', help='whether to holdout tasks instructions in training decoders')
    parser.add_argument('--job_index', type=int, help='for use with slurm sbatch script, indexes the combination of seed and holdout tasks along with the model')
    args = parser.parse_args()

    MODEL_FOLDER = args.folder
    os.environ['MODEL_FOLDER']=MODEL_FOLDER
    EXP_FOLDER =MODEL_FOLDER+'/'+args.exp+'_holdouts'

    jobs = make_training_jobs(args.exp, args.models, args.seeds, args.holdouts, args.job_index)
    print(jobs)
    for job in jobs: 
        _seed, model, holdouts = job

        # if 'gpt' in model: stream_data=True
        # else: stream_data=False

        stream_data = True


        if args.mode == 'pipeline': 
            from instructRNN.trainers.model_trainer import *
            run_pipeline(EXP_FOLDER, model, _seed, holdouts,overwrite=args.overwrite, use_checkpoint = args.use_checkpoint, ot = args.ot, stream_data=stream_data)      

        if args.mode == 'train': 
            from instructRNN.trainers.model_trainer import *
            train_model(EXP_FOLDER, model, _seed, holdouts, overwrite=args.overwrite, use_checkpoint = args.use_checkpoint)     

        if args.mode == 'tune': 
            from instructRNN.trainers.model_trainer import *
            tune_model(EXP_FOLDER, model, _seed, holdouts, overwrite=args.overwrite, use_checkpoint=args.use_checkpoint)     

        if args.mode == 'test': 
            from instructRNN.trainers.model_trainer import *
            test_model(EXP_FOLDER, model, _seed, holdouts, overwrite=args.overwrite)   

        if args.mode == 'context' or args.mode == 'c': 
            from instructRNN.trainers.context_trainer import *
            if not len(holdouts[1]): 
                tasks = TASK_LIST
            else: 
                tasks = list(holdouts[1])
            print(tasks)
            train_contexts(EXP_FOLDER, model, _seed, holdouts, args.layer, overwrite=args.overwrite, 
                                lr=0.02, num_contexts=100, tasks=tasks)

        if args.mode == 'decoder' or args.mode == 'd': 
            from instructRNN.trainers.decoder_trainer import *
            train_decoder(EXP_FOLDER, model, _seed, holdouts, args.use_holdouts, overwrite=args.overwrite)

