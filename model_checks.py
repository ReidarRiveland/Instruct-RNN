import itertools
import os
from instructRNN.trainers.model_trainer import *
from instructRNN.models.full_models import make_default_model
from instructRNN.analysis.model_analysis import get_model_performance, task_eval

from instructRNN.tasks.tasks import SWAPS_DICT, ALIGNED_DICT
from instructRNN.models.full_models import all_models, untuned_models, tuned_models

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help='folder where models and data will be stored')
    parser.add_argument('exp', help='type of experiment, refering to which holdout sets to use, must be \'swap\' or \'aligned\' ')
    parser.add_argument('--mode', default='perf', help='type of experiment, refering to which holdout sets to use, must be \'swap\' or \'aligned\' ')
    parser.add_argument('--models', default=all_models, nargs='*', help='list of model names to train, default is all models')
    parser.add_argument('--holdouts', type=int, default=range(9),  nargs='*', help='list of ints that index the holdout sets to use')
    parser.add_argument('--seeds', type=int, default=range(5), nargs='+', help='random seeds to use when training')
    args = parser.parse_args()

    if args.exp == 'swap': 
        holdout_dict = SWAPS_DICT
    elif args.exp =='aligned': 
        holdout_dict = ALIGNED_DICT


    MODEL_FOLDER = args.folder
    for model_name in args.models: 
        for holdout in args.holdouts: 
            holdout_file = args.exp+'_holdouts'
            load_folder = MODEL_FOLDER + '/'+ holdout_file+'/'+args.exp+str(holdout)+'/'+model_name
            for seed in args.seeds: 
                suffix = 'seed'+str(seed)
                if os.path.exists(load_folder+'/'+model_name+'_'+suffix+'.pt'):
                    print('loading model at ' + load_folder + ' for seed ' + str(seed)+ '\n')
                    model = make_default_model(model_name)
                    model.load_model(load_folder, suffix='_'+suffix)
                    if args.mode == 'perf':
                        perf = get_model_performance(model)
                        print(list(zip(TASK_LIST, perf)))
                    elif args.mode == 'holdout':                         
                        for task in holdout_dict[holdout_file]:
                            perf = task_eval(model, task, 256)
                            print((task, perf)+'\n')
                    elif args.mode == 'data_len':                         
                        data = pickle.load(open(load_folder+'/'+suffix+'_training_correct', 'rb'))
                        for task in TASK_LIST:
                            print(str(len(data[task])+'\n'))
                    else:
                        raise 'invalid mode type'

                else: 
                    print('no model found at ' + load_folder + ' for seed '+str(seed))
                    print(load_folder+'/'+model_name+suffix)
