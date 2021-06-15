from numpy.lib import utils
import torch
import numpy as np

from task import Task, construct_batch
from utils import isCorrect

import matplotlib.pyplot as plt
import seaborn as sns

task_list = Task.TASK_LIST

def get_model_performance(model, num_batches): 
    model.eval()
    batch_len = 128
    with torch.no_grad():
        perf_dict = dict.fromkeys(task_list)
        for task_type in task_list:
            for _ in range(num_batches): 
                mean_list = [] 
                ins, targets, _, target_dirs, _ = construct_batch(task_type, batch_len)
                task_info = model.get_task_info(batch_len, task_type)
                out, _ = model(task_info, ins)
                mean_list.append(np.mean(isCorrect(out, targets, target_dirs)))
            perf_dict[task_type] = np.mean(mean_list)
    return perf_dict 

def plot_trained_performance(self, model_list):
    barWidth = 0.1
    for i, model in enumerate(model_list):  
        perf_dict = self._get_performance(model, 5)
        values = list(perf_dict.values())
        len_values = len(task_list)
        if i == 0:
            r = np.arange(len_values)
        else:
            r = [x + barWidth for x in r]
        plt.bar(r, values, width =barWidth, label = model_list[i])

    plt.ylim(0, 1.15)
    plt.title('Trained Performance')
    plt.xlabel('Task Type', fontweight='bold')
    plt.ylabel('Percentage Correct')
    r = np.arange(len_values)
    plt.xticks([r + barWidth for r in range(len_values)], task_list)
    plt.legend()
    plt.show()

def plot_model_response(model, task_type, trials = None, trial_num = 0, instruct = None):
    model.eval()
    with torch.no_grad(): 
        if trials == None: 
            task = construct_batch(task_type, 1)

        tar = task.targets
        ins = task.inputs

        if instruct is not None: task_info = [instruct]
        else: task_info = model.get_task_info(ins.shape[0], task_type)
        
        out, hid = model(task_info, ins)

        correct = isCorrect(out, torch.Tensor(tar), task.target_dirs)[trial_num]
        out = out.detach().cpu().numpy()[trial_num, :, :]
        hid = hid.detach().cpu().numpy()[trial_num, :, :]

        try: 
            task_info_embedding = model.langModel(task_info).unsqueeze(1).repeat(1, ins.shape[1], 1)
        except: 
            task_info_embedding = task_info.unsqueeze(1).repeat(1, ins.shape[1], 1)

        fix = ins[trial_num, :, 0:1]            
        mod1 = ins[trial_num, :, 1:1+Task.STIM_DIM]
        mod2 = ins[trial_num, :, 1+Task.STIM_DIM:1+(2*Task.STIM_DIM)]

        to_plot = [fix.T, mod1.squeeze().T, mod2.squeeze().T, task_info_embedding.T, tar[trial_num, :, :].T, out.squeeze().T]
        gs_kw = dict(width_ratios=[1], height_ratios=[1, 5, 5, 2, 5, 5])
        ylabels = ['fix.', 'mod. 1', 'mod. 2', 'Task Info', 'Target', 'Response']

        fig, axn = plt.subplots(6,1, sharex = True, gridspec_kw=gs_kw)
        cbar_ax = fig.add_axes([.91, .3, .03, .4])
        for i, ax in enumerate(axn.flat):
            sns.heatmap(to_plot[i], yticklabels = False, cmap = 'Reds', ax=ax, cbar=i == 0, vmin=0, vmax=1, cbar_ax=None if i else cbar_ax)
            ax.set_ylabel(ylabels[i])
            if i == 0: 
                ax.set_title(task_type +' trial info; correct: ' + str(correct))
            if i == 5: 
                ax.set_xlabel('time')

        plt.show()



def _get_instruct_rep(self, instruct_dict, depth='full'):
    self.langModel.to(device)
    self.langModel.eval()
    with torch.no_grad(): 
        task_indices = []
        rep_tensor = torch.Tensor().to(device)
        for i, task in enumerate(instruct_dict.keys()):
            instructions = instruct_dict[task]
            if depth == 'full': 
                out_rep = self.langModel(instructions)
            elif depth == 'transformer': 
                tokens = self.langModel.model.tokenize(instructions)
                for key, value in tokens.items():
                    tokens[key] = value.to(device)
                sent_embedding = self.langModel.model(tokens)['sentence_embedding']
                out_rep = sent_embedding
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


def plot_embedding(self, dim=2, embedder = 'PCA', interm_dim = 20, depth = 'full', tasks = task_list, plot_avg = False, train_only=False, RGBY = False, cmap = matplotlib.cm.get_cmap('tab20')):
    assert dim in [2, 3], "embedding dimension must be 2 or 3"

    train_indices, train_rep = self._get_instruct_rep(train_instruct_dict, depth=depth)
    test_indices, test_rep = self._get_instruct_rep(test_instruct_dict, depth=depth)

    if plot_avg: 
        train_indices, train_rep = self._get_avg_rep(train_indices, train_rep)

    if len(test_rep.shape)>2: 
        test_rep = test_rep.squeeze()

    if embedder == 'PCA': 
        embedded_train = PCA(n_components=dim).fit_transform(train_rep)
        embedded_test = PCA(n_components=dim).fit_transform(test_rep)
    elif embedder == 'TSNE': 
        pca_reduced_train = PCA(n_components=interm_dim).fit_transform(train_rep)
        pca_reduced_test = PCA(n_components=interm_dim).fit_transform(test_rep)
        embedded_train = TSNE(n_components=dim).fit_transform(pca_reduced_train)
        embedded_test = TSNE(n_components=dim).fit_transform(pca_reduced_test)
    elif embedder == 'UMAP': 
        embedded_train = umap.UMAP().fit_transform(train_rep)
        embedded_test = umap.UMAP().fit_transform(test_rep)
        
    task_indices = [task_list.index(task) for task in tasks] 

    if tasks != task_list:
        trainindexrep = [rep for rep in zip(train_indices, embedded_train) if rep[0] in task_indices]
        testindexrep = [rep for rep in zip(test_indices, embedded_test) if rep[0] in task_indices]
        train_indices, embedded_train = zip(*trainindexrep)
        test_indices, embedded_test = zip(*testindexrep)
        embedded_train = np.stack(embedded_train)
        embedded_test = np.stack(embedded_test)


    color_train = np.array(train_indices).astype(int)
    color_test = np.array(test_indices).astype(int)

    if dim == 2: 
        fig, ax = plt.subplots(figsize=(6, 5))
        if RGBY == True: 
            colors = ['Green']*15 + ['Red']*15 + ['Yellow']*15 + ['Blue']*15
            jitter = np.random.uniform(0.0, 0.1, size=len(colors))
            plt.scatter(embedded_train[:, 0], embedded_train[:, 1] + jitter, color= colors, cmap=cmap, s=100) 
            if not train_only:
                test_colors =  ['Green']*5 + ['Red']*5 + ['Yellow']*5 + ['Blue']*5
                plt.scatter(embedded_test[:, 0], embedded_test[:, 1], color = test_colors, marker= "X", s=100)   
        # else: 
        #     plt.scatter(embedded_train[:, 0], embedded_train[:, 1], c=cmap(color_train), cmap=cmap, s=100)
        # if not train_only:
        #     plt.scatter(embedded_test[:, 0], embedded_test[:, 1], c=cmap(color_test), marker= "X", s=100)

        else: 
            plt.scatter(embedded_train[:, 0], embedded_train[:, 1], color=task_cmap(color_train), cmap=cmap, s=100)
        if not train_only:
            plt.scatter(embedded_test[:, 0], embedded_test[:, 1], color=task_cmap(color_test), marker= "X", s=100)
        plt.xlabel("PC 1", fontsize = 18)
        plt.ylabel("PC 2", fontsize = 18)
    else: 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embedded_train[:, 0], embedded_train[:, 1], embedded_train[:,2], color=task_cmap(color_train), cmap=cmap, s=100)
        if not train_only:
            ax.scatter(embedded_test[:, 0], embedded_test[:, 1], embedded_test[:, 2], color=task_cmap(color_test), marker= "X", s=100)
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')


    plt.suptitle(r'$\textbf{PCA of Instruction Embeddings}$', fontsize=14, fontweight='bold')
    plt.title('S-Bert (end-to-end)')
    digits = np.arange(len(tasks))
    if RGBY == True: 
        Patches = [mpatches.Patch(color=['Red', 'Green', 'Blue', 'Yellow'][i-task_indices[0]], label=task_list[i]) for i in task_indices]
    else: 
        Patches = [mpatches.Patch(color=task_cmap([i])[0], label=task_list[i]) for i in task_indices]
    if not train_only: 
        Patches.append(Line2D([0], [0], marker='X', color='w', label='test data', markerfacecolor='grey', markersize=10))
        Patches.append(Line2D([0], [0], marker='o', color='w', label='train data', markerfacecolor='grey', markersize=10))

    plt.legend(handles=Patches)
    plt.show()



