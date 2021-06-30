import torch
import numpy as np
from torch.nn.modules import transformer

from task import Task
from data import TaskDataSet
from utils import isCorrect
from data import TaskDataSet

task_list = Task.TASK_LIST

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def get_model_performance(model, num_batches): 
    model.eval()
    batch_len = 128
    with torch.no_grad():
        perf_dict = dict.fromkeys(task_list)
        for task in task_list:
            print(task)
            mean_list = [] 
            for _ in range(num_batches): 
                ins, targets, _, target_dirs, _ = next(TaskDataSet(num_batches=1, task_ratio_dict={task:1}).stream_batch())
                task_info = model.get_task_info(batch_len, task)
                out, _ = model(task_info, ins.to(model.__device__))
                mean_list.append(np.mean(isCorrect(out, targets, target_dirs)))
            perf_dict[task] = np.mean(mean_list)
    return perf_dict 

def get_instruct_reps(langModel, instruct_dict, depth='full'):
    langModel.eval()
    if depth=='transformer': 
        assert hasattr(langModel, 'transformer'), 'language model must be transformer to evaluate a that depth'
        rep_dim = 768
    else: rep_dim = langModel.out_dim 
    instruct_reps = torch.empty(len(instruct_dict.keys()), len(list(instruct_dict.values())[0]), rep_dim)
    with torch.no_grad():      
        for i, instructions in enumerate(instruct_dict.values()):
            if depth == 'full': 
                out_rep = langModel(list(instructions))
            elif depth == 'transformer': 
                out_rep = langModel.forward_transformer(list(instructions))
            instruct_reps[i, :, :] = out_rep
    return instruct_reps.cpu().numpy().astype(np.float64)


def get_task_reps(model, epoch='prep', num_trials =100):
    assert epoch in ['stim', 'prep'] or epoch.isnumeric(), "entered invalid epoch: %r" %epoch
    model.eval()
    with torch.no_grad(): 
        task_reps = np.empty((len(task_list), 100, model.hid_dim))
        for i, task in enumerate(task_list): 
            ins, targets, _, _, _ =  next(TaskDataSet(num_batches=1, batch_len=num_trials, task_ratio_dict={task:1}).stream_batch())

            task_info = model.get_task_info(num_trials, task)
            _, hid = model(task_info, ins.to(model.__device__))

            hid = hid.cpu().numpy()

            for j in range(num_trials): 
                if epoch.isnumeric(): epoch_index = int(epoch)
                if epoch == 'stim': epoch_index = np.where(targets.numpy()[j, :, 0] == 0.85)[0][-1]
                if epoch == 'prep': epoch_index = np.where(ins.numpy()[j, :, 1:]>0.25)[0][0]-1
                task_reps[i, j, :] = hid[j, epoch_index, :]
    return task_reps.astype(np.float64)


def get_hid_var_resp(model, task, trials, num_repeats = 10, task_info=None): 
    model.eval()
    with torch.no_grad(): 
        num_trials = trials.inputs.shape[0]
        total_neuron_response = np.empty((num_repeats, num_trials, 120, 128))
        for i in range(num_repeats): 
            if task_info is None: task_info = model.get_task_info(num_trials, task)
            _, hid = model(task_info, torch.Tensor(trials.inputs).to(model.__device__))
            hid = hid.cpu().numpy()
            total_neuron_response[i, :, :, :] = hid
        mean_neural_response = np.mean(total_neuron_response, axis=0)
    return total_neuron_response, mean_neural_response

def reduce_rep(reps, dim=2, reduction_method='PCA'): 
    if reduction_method == 'PCA': 
        embedder = PCA(n_components=dim)
    elif reduction_method == 'tSNE': 
        embedder = TSNE(n_components=2)

    embedded = embedder.fit_transform(reps.reshape(16*reps.shape[1], -1))

    if reduction_method == 'PCA': 
        explained_variance = embedder.explained_variance_ratio_
    else: 
        explained_variance = None

    return embedded.reshape(16, reps.shape[1], dim), explained_variance



if __name__ == "__main__":



    #tasks_to_plot = ['COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2']


    from sklearn.metrics.pairwise import cosine_similarity
    import seaborn as sns

    from rnn_models import InstructNet, SimpleNet
    from nlp_models import SBERT, BERT
    from data import TaskDataSet
    from utils import train_instruct_dict


    model = InstructNet(BERT(20, train_layers=['11']), 128, 1)
    #model = SimpleNet(128, 1)
    model.model_name+='_seed2'

    model.load_model('_ReLU128_14.6/single_holdouts/Go')
    model.to(torch.device(0))

    model.instruct_mode='validation'

    s_bert_model_perf 
    bert_model_perf = get_model_performance(model, 5)


    bert_model_perf

    s_bert_reps = get_task_reps(model)
    instruct_reps = get_instruct_reps(model.langModel, train_instruct_dict)


    #reps = get_task_reps(model)
    reps_reduced, var_explained = reduce_rep(reps, reduction_method='PCA')

    plot_RDM(np.mean(instruct_reps, axis=1), cmap=sns.color_palette("rocket_r", as_cmap=True))


    # from utils import train_instruct_dict
    # instruct_reps = get_instruct_reps(model.langModel, train_instruct_dict)

    # cmap = matplotlib.cm.get_cmap('tab20')
    # Patches = []
    # if dim ==3: 
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     scatter = [to_plot[:, 0], to_plot[:, 1], to_plot[:,2], cmap(task_indices), cmap, marker_size]
    #     ax.scatter(to_plot[:, 0], to_plot[:, 1], to_plot[:,2], c = cmap(task_indices), cmap=cmap, s=marker_size)
    #     ax.set_xlabel('PC 1')
    #     ax.set_ylabel('PC 2')
    #     ax.set_zlabel('PC 3')



    # for 


    # #plt.suptitle(r"$\textbf{PCA Embedding for Task Representation$", fontsize=18)
    # plt.title(Title)
    # digits = np.arange(len(tasks))
    # plt.tight_layout()
    # Patches = [mpatches.Patch(color=cmap(d), label=task_list[d]) for d in set(task_indices)]
    # scatter.append(Patches)
    # plt.legend(handles=Patches)
    # #plt.show()
    # return explained_variance, scatter

