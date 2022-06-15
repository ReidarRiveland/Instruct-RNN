from instructRNN.trainers.model_trainer import *
from instructRNN.trainers.decoder_trainer import *
from instructRNN.trainers.context_trainer import *
from instructRNN.tasks.tasks import SWAPS_DICT, ALIGNED_DICT
from instructRNN.models.full_models import all_models


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help='folder where models and data will be stored')
    parser.add_argument('exp', help='type of experiment, refering to which holdout sets to use, must be \'swap\' or \'aligned\' ')
    parser.add_argument('--mode', default='train', help='training mode to use, must be \'train\', \'tune\', \'test\', \'decoder\' ( \'d\'),\'context\' ( \'c\')')
    parser.add_argument('--models', default=all_models, nargs='*', help='list of model names to train, default is all models')
    parser.add_argument('--holdouts', type=int, default=None,  nargs='*', help='list of ints that index the holdout sets to use')
    parser.add_argument('--overwrite', default=False, action='store_true', help='whether or not to overwrite existing files')
    parser.add_argument('--seeds', type=int, default=[0], nargs='+', help='random seeds to use when training')
    parser.add_argument('--layer', default='last', help='the dim corresponding to the layer the contexts gets trained at, \
                                                    must be emd or last, only for use if mode is context')
    parser.add_argument('--use_holdouts', default=False, action='store_true', help='whether to holdout tasks instructions in training decoders')
    args = parser.parse_args()

    MODEL_FOLDER = args.folder
    EXP_FOLDER =MODEL_FOLDER+'/'+args.exp+'_holdouts'

    if args.exp == 'swap': 
        _holdout_dict = SWAPS_DICT
    elif args.exp == 'algined': 
        _holdout_dict = ALIGNED_DICT

    if args.holdouts is None: 
        holdout_dict = _holdout_dict
    else: 
        holdout_dict = dict([list(_holdout_dict.items())[i] for i in args.holdouts])

    if args.mode == 'train': 
        train_model_set(EXP_FOLDER, args.models, args.seeds, holdout_dict, overwrite=args.overwrite)     
    if args.mode == 'tune': 
        tune_model_set(EXP_FOLDER, args.models, args.seeds, holdout_dict, overwrite=args.overwrite)     
    if args.mode == 'test': 
        test_model_set(EXP_FOLDER, args.models, args.seeds, holdout_dict, overwrite=args.overwrite)   
    if args.mode == 'context' or args.mode == 'c': 
        train_context_set(EXP_FOLDER, args.models, args.seeds, holdout_dict, args.layer, overwrite=args.overwrite, lr=0.005, num_contexts=5, tasks=TASK_LIST[::-1][5:])
    if args.mode == 'decoder' or args.mode == 'd': 
        train_decoder_set(EXP_FOLDER, args.models, args.seeds, holdout_dict, args.layer, args.use_holdouts, overwrite=args.overwrite)
