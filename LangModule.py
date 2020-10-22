import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import math
import numpy as np
import random

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.ndimage.filters import gaussian_filter1d
import pickle

from Task import Task
task_list = Task.TASK_LIST

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

train_instruct_dict = pickle.load(open('Instructions/train_instruct_dict', 'rb'))
test_instruct_dict = pickle.load(open('Instructions/test_instruct_dict', 'rb'))

swaps= [['Go', 'Anti DM'], ['Anti RT Go', 'DMC'], ['COMP2', 'RT Go']]
swapped_task_list = ['Anti DM', 'COMP2', 'Anti Go', 'DMC', 'DM', 'Go', 'MultiDM', 'Anti MultiDM', 'COMP1', 'RT Go', 'MultiCOMP1', 'MultiCOMP2', 'DMS', 'DNMS', 'Anti RT Go', 'DNMC']
instruct_swap_dict = dict(zip(swapped_task_list, train_instruct_dict.values()))

PAD_LEN = 25

def toNumerals(tokenizer, instructions): 
    ins_temp = []
    for instruct in instructions:
        embedding = torch.ones((PAD_LEN, 1))
        tokenized = torch.Tensor(tokenizer.encode(instruct)).unsqueeze(1)
        embedding[:tokenized.shape[0]] = tokenized
        ins_temp.append(embedding)
    return torch.stack(ins_temp).squeeze().long().to(device)

def get_batch(batch_size, tokenizer, task_type = None, instruct_mode = None):
    assert instruct_mode in [None, 'instruct_swap', 'shuffled']
    batch = []
    batch_target_index = []
    for i in range(batch_size):
        task = task_type
        if task is None: 
            task = np.random.choice(task_list)
        if instruct_mode == 'instruct_swap': 
            instruct_dict = instruct_swap_dict
        else: 
            instruct_dict = train_instruct_dict
        instruct = random.choice(instruct_dict[task])
        batch_target_index.append(task_list.index(task))
        if instruct_mode == 'shuffled': 
            instruct = instruct.split()
            shuffled = np.random.permutation(instruct)
            instruct = ' '.join(list(shuffled))

        batch.append(instruct)
    if tokenizer is not None: 
        batch = toNumerals(tokenizer, batch)
    return batch, batch_target_index


class LangModule(): 
    def __init__(self, langModel, foldername = '', instruct_mode = None): 
        self.langModel = langModel
        self.embedderStr = langModel.embedderStr
        if len(foldername) > 0: foldername = foldername+'/'
        self.loss_list = []
        self.val_loss_list = []
        self.instruct_mode = instruct_mode
        self.model_classifier = nn.Sequential(self.langModel, nn.Linear(self.langModel.out_dim, len(task_list), nn.ReLU()))
        self.shuffled = False
        self.classifier_criterion = nn.CrossEntropyLoss()

        self.foldername = foldername
        self.filename = self.embedderStr + '_' + str(self.langModel.out_dim)

    def train_classifier(self, batch_len, num_batches, epochs, optim_method = 'adam', lr=0.001, weight_decay=0, shuffle = False, train_out_only = False):
        self.shuffled = shuffle
        if optim_method == 'adam': 
            opt = optim.Adam(self.model_classifier.parameters(), lr, weight_decay=weight_decay)
        if optim_method == 'SGD': 
            opt = optim.SGD(self.model_classifier.parameters(), lr, weight_decay=weight_decay)
        if train_out_only: 
            opt = optim.Adam(self.model_classifier.linear.parameters(), lr, weight_decay=weight_decay)
        self.identity = nn.Identity()

        best_val_loss = 1e5
        self.model_classifier.to(device)
        self.model_classifier.train()
        for i in range(epochs):
            for j in range(num_batches): 
                opt.zero_grad()
                ins_temp, targets = get_batch(batch_len, tokenizer = self.langModel.tokenizer, instruct_mode = self.instruct_mode)
                tensor_targets = torch.Tensor(targets).to(device)
                out = self.model_classifier(ins_temp)
                loss = self.classifier_criterion(out, tensor_targets.long())
                loss.backward()
                opt.step()
                if j%100 == 0: 
                    print(j, ':', loss.item())
                self.loss_list.append(loss.item())

                val_loss = self.get_val_loss()
                self.val_loss_list.append(val_loss)

                if val_loss < best_val_loss: 
                    best_val_loss = val_loss
                    torch.save(self.model_classifier.state_dict(), self.foldername+ 'LanguageModels/'+self.filename+'.pt')
        self.model_classifier.load_state_dict(torch.load(self.foldername+'LanguageModels/'+ self.filename+'.pt'))

    def loadLangModel(self): 
        self.model_classifier.load_state_dict(torch.load(self.foldername+'/LanguageModels/'+ self.filename+'.pt'))
    
    def save_classifier_training_data(self): 
        pickle.dump(self.loss_list, open(self.foldername+'LanguageModels/'+self.filename+'_training_loss', 'wb'))
        pickle.dump(self.val_loss_list, open(self.foldername+'LanguageModels/'+self.filename+'_val_loss', 'wb'))

    def load_classifier_training_data(self): 
        self.loss_list = pickle.load(open(self.foldername+'LanguageModels/'+self.filename+'_training_loss', 'wb'))
        self.val_loss_list = pickle.load(open(self.foldername+'/LanguageModels/'+self.filename+'_val_loss', 'wb'))


    def get_val_loss(self): 
        total_loss = 0
        num_sentences = 0
        self.model_classifier.eval()
        for i, task in enumerate(task_list): 
            instructions = test_instruct_dict[task]
            if self.langModel.tokenizer is not None: 
                instructions = toNumerals(self.langModel.tokenizer, instructions)
            out = self.model_classifier(instructions)
            tensor_targets = torch.full((len(instructions), ), i).to(device)
            loss = self.classifier_criterion(out, tensor_targets.long())            
            total_loss += loss.item()
        self.model_classifier.train()
        return total_loss/len(task_list)

    def plot_loss(self, mode):
        assert mode in ['train', 'validation'], 'mode must be "train" or "validation"'
        if mode == 'train':
            plt.plot(self.loss_list, label = self.embedderStr)
            plt.legend()
            plt.title('Training Loss')
        if mode == 'validation': 
            plt.plot(self.val_loss_list, label = self.embedderStr)
            plt.legend()
            plt.title('Validation Loss')
        plt.ylabel('Cross Entropy Loss')
        plt.xlabel('total mini-batches')
        plt.show()

    def _test_performance(self): 
        self.model_classifier.eval()
        perf_dict = {}
        num_task = len(task_list)
        confuse_mat = np.zeros((num_task, num_task), dtype=int)
        for i, task in enumerate(task_list): 
            num_correct = 0
            instructions = test_instruct_dict[task]
            if self.langModel.tokenizer is not None: 
                instructions = toNumerals(self.langModel.tokenizer, instructions)
            model_cat = torch.argmax(self.model_classifier(instructions), dim=1).cpu().numpy()
            for j in model_cat: 
                confuse_mat[i, j] += 1
                if j == i:
                    num_correct += 1  
            perf_dict[task] = num_correct/(j+1)
        return perf_dict, confuse_mat

    def plot_confusion_matrix(self): 
        _, confuse_mat = self._test_performance()
        f, ax = plt.subplots(1,1)
        ax = sns.heatmap(confuse_mat, ax = ax, yticklabels = task_list, xticklabels= task_list, cbar = False, annot=True, fmt="d")
        f.suptitle('Confusion Matrix; embedder: {}'.format(self.embedderStr))
        plt.show()

    def _get_instruct_rep(self, instruct_dict):
        self.langModel.to(device)
        self.langModel.eval()
        with torch.no_grad(): 
            task_indices = []
            rep_tensor = torch.Tensor().to(device)
            for i, task in enumerate(instruct_dict.keys()):
                instructions = instruct_dict[task]
                if self.langModel.tokenizer is not None: 
                    instructions = toNumerals(self.langModel.tokenizer, instructions)
                out_rep = self.langModel(instructions)
                task_indices += ([i]*len(instructions))
                rep_tensor = torch.cat((rep_tensor, out_rep), dim=0)
        return task_indices, rep_tensor.cpu().detach().numpy()

    def _get_avg_rep(self, task_indices, reps): 
        avg_rep_list = []
        for i in set(task_indices):
            avg_rep = np.zeros(reps.shape[-1])
            for index, rep in zip(task_indices, reps):
                if i == index: 
                    avg_rep += rep
            avg_rep_list.append(avg_rep/task_indices.count(i))

        task_set = list(set(task_indices))
        avg_reps = np.array(avg_rep_list)
        
        return task_set, avg_reps


    def plot_embedding(self, dim=2, tasks = task_list, plot_avg = False, train_only=False):
        assert dim in [2, 3], "embedding dimension must be 2 or 3"

        train_indices, train_rep = self._get_instruct_rep(train_instruct_dict)
        test_indices, test_rep = self._get_instruct_rep(test_instruct_dict)

        if plot_avg: 
            train_indices, train_rep = self._get_avg_rep(train_indices, train_rep)

        if len(test_rep.shape)>2: 
            test_rep = test_rep.squeeze()

        embedded_train = PCA(n_components=dim).fit_transform(train_rep)
        embedded_test = PCA(n_components=dim).fit_transform(test_rep)

        task_indices = [task_list.index(task) for task in tasks] 

        if tasks != task_list:
            trainindexrep = [rep for rep in zip(train_indices, embedded_train) if rep[0] in task_indices]
            testindexrep = [rep for rep in zip(test_indices, embedded_test) if rep[0] in task_indices]
            train_indices, embedded_train = zip(*trainindexrep)
            test_indices, embedded_test = zip(*testindexrep)
            embedded_train = np.stack(embedded_train)
            embedded_test = np.stack(embedded_test)

        cmap = matplotlib.cm.get_cmap('tab20')

        color_train = np.array(train_indices).astype(int)
        color_test = np.array(test_indices).astype(int)

        if dim == 2: 
            fig, ax = plt.subplots(figsize=(12, 10))
            plt.scatter(embedded_train[:, 0], embedded_train[:, 1], c=cmap(color_train), cmap=cmap, s=100)
            if not train_only:
                plt.scatter(embedded_test[:, 0], embedded_test[:, 1], c=cmap(color_test), marker= "X", s=100)
            plt.setp(ax, xticks=[], yticks=[])
            plt.xlabel("PC 1", fontsize = 18)
            plt.ylabel("PC 2", fontsize = 18)
        else: 
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(embedded_train[:, 0], embedded_train[:, 1], embedded_train[:,2], c=cmap(color_train), cmap=cmap, s=100)
            if not train_only:
                ax.scatter(embedded_test[:, 0], embedded_test[:, 1], embedded_test[:, 2], c=cmap(color_test), marker= "X", s=100)
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
            ax.set_zlabel('PC 3')


        plt.title("PCA Embedding for Distributed Rep.", fontsize=18)
        digits = np.arange(len(tasks))
        Patches = [mpatches.Patch(color=cmap(i), label=task_list[i]) for i in task_indices]
        if not train_only: 
            Patches.append(Line2D([0], [0], marker='X', color='w', label='test data', markerfacecolor='grey', markersize=10))
        Patches.append(Line2D([0], [0], marker='o', color='w', label='train data', markerfacecolor='grey', markersize=10))

        plt.legend(handles=Patches)
        plt.show()

def plot_lang_perf(mod_dict, mode, smoothing): 
    COLOR_DICT = {'Model1': 'blue', 'SIF':'brown', 'BoW': 'orange', 'gpt': 'red', 'BERT': 'green', 'SBERT': 'purple', 'InferSent':'yellow', 'Transformer': 'pink'}
    if mode == 'train': 
        for label, mod in mod_dict.items(): 
            loss = smoothed_perf = gaussian_filter1d(mod.loss_list, sigma=smoothing)
            plt.plot(loss, label = label, color = COLOR_DICT[mod.embedderStr])
        plt.legend()
        plt.show()
        plt.title('Training Loss')
    if mode == 'validation': 
        for label, mod in mod_dict.items(): 
            loss = smoothed_perf = gaussian_filter1d(mod.val_loss_list, sigma=smoothing)
            min_loss = np.min(loss)
            p = plt.plot(loss, label = label)
            plt.hlines(min_loss, 0, len(loss), colors = p[-1].get_color(), linestyles='dotted')
        plt.legend()
        plt.show()
        plt.title('Validation Loss')
    plt.ylabel('Cross Entropy Loss')
    plt.xlabel('total mini-batches')
