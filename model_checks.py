import itertools
import os
from instructRNN.trainers.model_trainer import *
from instructRNN.models.full_models import make_default_model
from instructRNN.analysis.model_analysis import get_model_performance

from instructRNN.tasks.tasks import SWAPS_DICT, ALIGNED_DICT
from instructRNN.models.full_models import all_models, untuned_models, tuned_models

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help='folder where models and data will be stored')
    parser.add_argument('exp', help='type of experiment, refering to which holdout sets to use, must be \'swap\' or \'aligned\' ')
    parser.add_argument('--models', default=untuned_models, nargs='*', help='list of model names to train, default is all models')
    parser.add_argument('--holdouts', type=int, default=range(9),  nargs='*', help='list of ints that index the holdout sets to use')
    parser.add_argument('--seeds', type=int, default=range(5), nargs='+', help='random seeds to use when training')
    args = parser.parse_args()

    MODEL_FOLDER = args.folder
    for model_name in args.models: 
        for holdout in args.holdouts: 
            load_folder = MODEL_FOLDER + '/'+ args.exp+'_holdouts/'+args.exp+str(holdout)+'/'+model_name
            for seed in args.seeds: 
                suffix = '_seed'+str(seed)
                if os.path.exists(load_folder+'/'+model_name+suffix):
                    print('loading model at ' + load_folder + ' for seed ' + str(seed)+ '\n')
                    model = make_default_model(model_name)
                    model.load_model(load_folder, suffix=suffix)
                    perf = get_model_performance(model)
                    print(list(zip(TASK_LIST, perf)))
                else: 
                    print('no model found at ' + load_folder + ' for seed '+str(seed))
