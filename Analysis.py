import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#from Plotting import plot_all_holdout_curves, plot_learning_curves
from LangModule import LangModule, swaps
from NLPmodels import GPT, BERT, SBERT, BoW
from RNNs import instructNet, simpleNet
from jitRNNs import scriptSimpleNet 
import torch
import torch.nn as nn

from CogModule import CogModule, isCorrect
from Data import make_data
from Task import Task
task_list = Task.TASK_LIST


##Model training loop
epochs = 25
init_lr = 0.001
milestones = [5, 10, 15, 20]


foldername = '_ReLU128_19.5'

retrain_list = [

                ('RT Go', '_seed0', ['S-Bert train']), 
                ('RT Go', '_seed4', ['BERT train']), 

                ('Anti Go', '_seed1', ['S-Bert train']), 
                ('Anti Go', '_seed2', ['S-Bert train']), 
                ('Anti Go', '_seed4', ['S-Bert train']), 
                ('Anti Go', '_seed3', ['BERT train']), 

                ('MultiDM', '_seed2', ['S-Bert train']), 

                ('Anti DM', '_seed0', ['S-Bert train']),  
                ('Anti DM', '_seed3', ['S-Bert train']),  
                ('Anti DM', '_seed2', ['BERT train']),  

                ('COMP2', '_seed2', ['S-Bert train']), 

                ('DMC', '_seed4', ['S-Bert train'])]




for holdout, seed, models in retrain_list:
    print(holdout, seed, models) 
    model_dict = {}
    if 'S-Bert train' in models: 
        model_dict['S-Bert train'+seed] = instructNet(LangModule(SBERT(20)), 128, 1, 'relu', tune_langModel=True, langLayerList=['layer.11'])
    elif 'BERT train' in models: 
        model_dict['BERT train'+seed] = instructNet(LangModule(BERT(20)), 128, 1, 'relu', tune_langModel=True, langLayerList=['layer.11'])
    else: 
        pass
    
    cog = CogModule(model_dict)
    print(cog.model_dict.keys())

    try: 
        cog.load_training_data(holdout, foldername, seed)
    except: 
        pass
    
    holdout_data = make_data(holdouts=[holdout], batch_size=128)

    cog.train(holdout_data, epochs, lr=init_lr, milestones = milestones, weight_decay=0.0)
    cog.save_models(holdout, foldername, seed)

holdout = 'Anti DM'
seed = '_seed4'
model_dict = {}
model_dict['S-Bert train'+seed] = instructNet(LangModule(SBERT(20)), 128, 1, 'relu', tune_langModel=True, langLayerList=['layer.11'])
model_dict['BERT'+seed] = instructNet(LangModule(BERT(20)), 128, 1, 'relu', tune_langModel=True, langLayerList=['layer.11'])
model_dict['BoW'+seed] = instructNet(LangModule(BoW()), 128, 1, 'relu', tune_langModel=False)
model_dict['Model1'+seed] = simpleNet(81, 128, 1, 'relu')
cog = CogModule(model_dict)
holdout_data = make_data(holdouts=[holdout], batch_size=128)
cog.train(holdout_data, epochs, lr=init_lr, milestones = milestones, weight_decay=0.0)
cog.save_models(holdout, foldername, seed)





foldername = '_ReLU128_19.5'

seeds = list(np.arange(5))
for i in [0]: 
    seed = '_seed'+str(i)
    for holdout in ['Anti DM']:
        model_dict = {}
        model_dict['S-Bert train'+seed] = instructNet(LangModule(SBERT(20)), 128, 1, 'relu', tune_langModel=True, langLayerList=['layer.11'])
        #model_dict['S-Bert'+seed] = instructNet(LangModule(SBERT(20)), 128, 1, 'relu', tune_langModel=False)
        # model_dict['BERT'+seed] = instructNet(LangModule(BERT(20)), 128, 1, 'relu', tune_langModel=False)
        # model_dict['GPT'+seed] = instructNet(LangModule(GPT(20)), 128, 1, 'relu', tune_langModel=False)

        # model_dict['BoW'+seed] = instructNet(LangModule(BoW()), 128, 1, 'relu', tune_langModel=False)
        # model_dict['Model1'+seed] = simpleNet(81, 128, 1, 'relu')
        cog = CogModule(model_dict)
        
        try: 
            cog.load_training_data(holdout, foldername, seed)
        except: 
            pass

        if holdout == 'Multitask':
            holdout_data = make_data(batch_size=128)
        else:
            holdout_data = make_data(holdouts=[holdout], batch_size=128)

        cog.train(holdout_data, epochs, lr=init_lr, milestones = milestones, weight_decay=0.0)
        cog.save_models(holdout, foldername, seed)




foldername = '_ReLU128_19.5'
model_dict = {}
seed = '_seed2'
model_dict['S-Bert linear'+seed] = instructNet(LangModule(SBERT(20, output_nonlinearity=nn.Identity())), 128, 1, 'relu', tune_langModel=True, langLayerList=['layer.11'])
#model_dict['S-Bert'+seed] = instructNet(LangModule(SBERT(20)), 128, 1, 'relu', tune_langModel=False)
# model_dict['BERT'+seed] = instructNet(LangModule(BERT(20)), 128, 1, 'relu', tune_langModel=False)
# model_dict['GPT'+seed] = instructNet(LangModule(GPT(20)), 128, 1, 'relu', tune_langModel=False)

# model_dict['BoW'+seed] = instructNet(LangModule(BoW()), 128, 1, 'relu', tune_langModel=False)
# model_dict['Model1'+seed] = simpleNet(81, 128, 1, 'relu')
cog = CogModule(model_dict)
cog.load_models('Anti RT Go', foldername)
cog._plot_trained_performance()


import pickle
###Holdout training loop 
for i in [0]:
    seed = '_seed'+str(i)
    modelSBERT_name = 'S-Bert train'+seed
    modelBERT_name = 'BERT train'+seed
    modelBOW_name = 'BoW'+seed
    model1_name = 'Model1'+seed

    for holdout in task_list:

        correct_dict = {key : np.zeros(100) for key in [modelSBERT_name, modelBERT_name, modelBOW_name, model1_name]}
        loss_dict = correct_dict.copy()

        model_dict = {}
        model_dict[modelSBERT_name] = instructNet(LangModule(SBERT(20)), 128, 1, 'relu', tune_langModel=True, langLayerList=['layer.11'])
        # model_dict[modelBERT_name] = instructNet(LangModule(BERT(20)), 128, 1, 'relu', tune_langModel=True, langLayerList=['layer.11'])
        # model_dict[modelBOW_name] = instructNet(LangModule(BoW()), 128, 1, 'relu', tune_langModel=False)
        # model_dict[model1_name] = simpleNet(81, 128, 1, 'relu')

        cog = CogModule(model_dict)


        for i in range(2): 

            print('Pass ' + str(i))
            
            holdout_data = make_data(task_dict = {holdout:1}, num_batches=100, batch_size=256)
            cog.load_models(holdout, foldername)



            cog.train(holdout_data, 1, lr=0.001)
            cog.sort_perf_by_task()
            for model_name in cog.model_dict.keys():
                correct_dict[model_name]+=np.round(np.array(cog.total_correct_dict[model_name])/5, 2)
                loss_dict[model_name]+= np.round(np.array(cog.total_loss_dict[model_name])/5, 2)

            cog.reset_data()

        holdout_name = holdout.replace(' ', '_')

        pickle.dump(correct_dict, open(foldername+'/'+holdout_name+'/'+seed+'_holdout_training_correct_dict', 'wb'))
        pickle.dump(loss_dict, open(foldername+'/'+holdout_name+'/'+seed+'_holdout_training_loss_dict', 'wb'))

        #cog.save_training_data(holdout_name, foldername, seed+'_holdout')



import pickle
task_sorted_correct = pickle.load(open(foldername+'/Go/_seed4_holdout_training_correct_dict', 'rb'))
task_sorted_correct