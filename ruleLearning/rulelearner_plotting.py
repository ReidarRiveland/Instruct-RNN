from instructRNN.tasks.tasks import TASK_LIST

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def plot_memory_trace(memory, ytics):
    sims = cosine_similarity(np.array(memory.mem_trace), memory.memory_encodings)
        
    fig, axn = plt.subplots(2,1, sharex=True, figsize =(8, 4))
    res = sns.heatmap(sims.T, xticklabels=[], yticklabels=ytics, ax=axn[0], cbar=False)
    res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 3)
    axn[1].plot(np.tile(memory.betas, len(memory.recalls)))
    plt.show()

def plot_task_learning(task, task_rewards, values, rpes, comp_reps, hard_repeat=5, embedding_perfs=None): 
    fig, axn = plt.subplots(3,1, figsize =(16, 12), gridspec_kw={'height_ratios':[3,3,3]})
    labels = ['estimated values', 'experienced returns']
    axn[0].plot(values)
    if embedding_perfs is not None:
        axn[0].plot(np.array(embedding_perfs))
        labels.append( 'true embedding perforamnce')
    axn[0].plot(task_rewards.reshape(-1, hard_repeat).mean(1), linewidth=0.5, alpha=0.8)
    axn[0].legend(labels=labels, prop = { "size": 8 }, loc='lower right')
    axn[0].set_ylabel('Prop. Correct', fontsize='medium')
    axn[0].tick_params(axis='y', which='major', labelsize=8)
    axn[0].set_ylim(0.0, 1.05)
    axn[0].set_xlim(0.0, values.shape[0])
    axn[0].set_xticklabels([])
    axn[0].set_xticks(np.linspace(0, comp_reps.shape[0], 9))
    plt.suptitle('Learning Dynamics for '+str(task))

    # axn[1].plot(rpes, linewidth=0.8, color='red')
    # axn[1].set_ylabel('RPE', fontsize='medium')
    # axn[1].tick_params(axis='y', which='major', labelsize=8)
    # axn[1].set_xlim(0.0, values.shape[0])
    # axn[1].set_xticklabels([])
    # axn[1].set_xticks(np.linspace(0, comp_reps.shape[0], 9))

    moving_avg = moving_average(task_rewards[:task_rewards.shape[0]], 20)
    axn[1].plot(moving_avg, color='green')
    axn[1].hlines(np.mean(moving_avg[-int(task_rewards.shape[0]/4):]), 0, len(moving_avg), linestyle='--')

    axn[1].tick_params(axis='y', which='major', labelsize=8)
    axn[1].tick_params(axis='x', which='major', labelsize=8)
    axn[1].set_ylim(0.0, 1.05)
    axn[1].set_xlim(0.0, task_rewards.shape[0])
    axn[1].set_xticklabels([])

    axn[1].set_ylabel('Moving Avg. Perf.', fontsize='medium')

    res = sns.heatmap(comp_reps.T, yticklabels=[], xticklabels=[], cbar=False, ax=axn[2])
    res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 8)
    axn[2].set_ylabel('Comp. Reps.', fontsize='medium')
    axn[2].set_xticks(np.linspace(0, comp_reps.shape[0], 9))

    axn[2].set_xticklabels(range(0, 900, 100))


    # res = sns.heatmap(task_inhbition_weight.T, yticklabels=[], xticklabels=[], vmax=1.0, vmin=0.0, cbar=False, ax=axn[4])
    # res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 8)
    # axn[4].set_ylabel('Task Recall Inhibition', fontsize='medium')

    # res = sns.heatmap(pc_inhbition_weight.T, yticklabels=[], cbar=False, vmax=1.0, vmin=0.0, ax=axn[5])
    # res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 8)
    # axn[5].set_ylabel('PC Recall Inhibition', fontsize='medium')

    # res.set_xticks(np.linspace(0, values.shape[0], 11))
    # res.set_xticklabels(np.linspace(0, task_rewards.shape[0], 11).astype(int), fontsize = 8, rotation='90')
    # axn[5].set_xlabel('Trials', fontsize='medium')

    plt.show()

def plot_population_learning(population_learning_data, task, axn=None, plot_individuals=True, **title_kwargs):
    if axn is None: 
        fig, axn = plt.subplots(1,1, figsize =(12, 8))
        handles = [Line2D([0], [0], label='Population Avg.', color='blue')]
        plt.legend(handles=handles)
        axn.set_xlabel('Trials')
        axn.set_ylabel('Task Performance')

    if plot_individuals:
        for i in range(population_learning_data.shape[0]):
            axn.plot(moving_average(population_learning_data[i, :], 50), alpha=0.6, linewidth=0.5)

    num_trials = population_learning_data.shape[-1]
    pop_avg = moving_average(population_learning_data.mean(0), 50)
    axn.plot(pop_avg, color='blue', linewidth=1.0)
    axn.set_title(task, **title_kwargs)
    
    axn.set_ylim(0.0, 1.05)
    axn.set_xlim(0.0, len(pop_avg)+10)

    axn.set_xticks(np.linspace(0, len(pop_avg), 9))
    axn.set_xticklabels(range(0, 900, 100))

    axn.hlines(np.mean(population_learning_data[:, -int(num_trials/4):]), 0, len(pop_avg), linestyle='--')

def plot_all_learning_curves(scores_path):
    fig, axn = plt.subplots(5,10, figsize =(24, 12), sharex=True, sharey=True)
    for i, task in enumerate(TASK_LIST): 
        reward_results = np.load(scores_path+'/'+task+'_rulelearning.npz')['rewards']
        if np.mean(reward_results[-int(reward_results.shape[0]/4):])>0.75:
            plot_population_learning(reward_results, task, axn.flatten()[i], color='Green')
        else: 
            plot_population_learning(reward_results, task, axn.flatten()[i])

    handles = [Line2D([0], [0], label='Population Avg.', color='blue'), Line2D([0], [0], label='Individuals', color='Green', alpha=0.6)]
    plt.legend(handles=handles)
    plt.suptitle('Population Learning Curves')
    plt.tight_layout()
    plt.show()


def plot_all_rulelearning_lolli_v(model_list, seed_num, num_clusters, marker = 'o', **kwargs):
    fig, axn = plt.subplots(1, 1, sharey = True, sharex=True, figsize =(11, 4))

    width = 1/(len(model_list)+1)
    ind = np.arange(len(SUBTASKS_DICT['small'][:-4]))

    axn.set_xticks(ind)
    axn.set_xticklabels('')
    axn.tick_params(axis='x', which='minor', bottom=False)
    axn.set_xticks(ind+0.75, minor=True)
    axn.set_xticklabels(SUBTASKS_DICT['small'][:-4], fontsize=6, minor=True, rotation=45, ha='right', fontweight='bold') 
    axn.set_xlim(-0.15, len(ind))

    axn.set_yticks(np.linspace(0, 1, 11))
    axn.set_yticklabels([f'{x:.0%}' for x in np.linspace(0, 1, 11)], fontsize=8)
    axn.set_ylim(0.0, 1.01)
    axn.set_ylabel('Percent Correct', size=8, fontweight='bold')

    for i, model_name in enumerate(model_list):  
        color = MODEL_STYLE_DICT[model_name][0]  
        data = RuleLearningData(model_name, seed_num, num_clusters)
        asym_perf = data.get_asym_perf().mean(-1)

        avg_perf = asym_perf.mean(-1)
        std = asym_perf.std(-1)

        x_mark = (ind+(width/2))+(i*width)
        axn.scatter(x_mark,  avg_perf, color=color, s=3, marker=marker)
        axn.vlines(x_mark, ymin=avg_perf-std, ymax=np.minimum(np.ones_like(avg_perf), avg_perf+std), color=color, linewidth=0.8, **kwargs)
        axn.vlines(x_mark, ymin=0, ymax=avg_perf, color=color, linewidth=0.5, alpha=0.5, **kwargs)

        #axn.axhline(np.mean(avg_perf), color=color, linewidth=1.0, alpha=0.8, zorder=0)

    #fig.legend(labels=[MODEL_STYLE_DICT[model_name][2] for model_name in model_list], loc=5, title='Models', title_fontsize = 'x-small', fontsize='x-small')        

    plt.tight_layout()
    return fig, axn