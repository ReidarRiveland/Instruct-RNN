import gensim
import gensim.downloader as api
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

from fse.models import SIF
from fse import IndexedList

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.nn.utils.rnn import pad_sequence

import math
import numpy as np
import random
from collections import defaultdict

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
PAD_LEN = 20

from transformers import GPT2Tokenizer, BertTokenizer
gptTokenizer = GPT2Tokenizer.from_pretrained('gpt2')
bertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_instruct_dict = pickle.load(open('Instructions/train_instruct_dict', 'rb'))
test_instruct_dict = pickle.load(open('Instructions/test_instruct_dict', 'rb'))

swapped_task_list = ['Anti DM', 'Anti MultiDM', 'Anti Go', 'DMS', 'DNMC', 'Go', 'MultiDM', 'RT Go', 'DNMS', 'DMC', 'DM', 'Anti RT Go']
instruct_swap_dict = dict(zip(swapped_task_list, train_instruct_dict.values()))


def shuffle_instruct_dict(instruct_dict): 
    shuffled_dict = defaultdict(list)
    for task_type, sentences in instruct_dict.items(): 
        for instruction in sentences:
            instruction = instruction.split()
            shuffled = np.random.permutation(instruction)
            shuffled_string = ' '.join(list(shuffled))
            shuffled_dict[task_type].append(shuffled_string)
    return shuffled_dict

def wordFreq():
    freq_dict = {}
    all_words = []
    all_sentences = []
    for task_type in task_list:
        instructions = train_instruct_dict[task_type] + test_instruct_dict[task_type]
        for sentence in instructions: 
            all_sentences.append(sentence)
            for word in sentence.split(): 
                all_words.append(word)
    for word in set(all_words):
        freq_dict[word] = all_words.count(word)
    
    split_sentences = []
    for sentence in all_sentences:
        list_sent = sentence.split()
        split_sentences.append(list_sent)

    return freq_dict, sorted(list(set(all_words))), all_sentences, split_sentences

def load_word_embedder_dict(splits): 
    word_embedder_dict = {}
    _,_,_, split_sentences = wordFreq()
    word_embedder_dict['w2v50'] = (KeyedVectors.load("wordVectors/w2v50.kv", mmap='r'), 50)
    word_embedder_dict['w2v100'] = (KeyedVectors.load("wordVectors/w2v100.kv", mmap='r'), 100)
    word_embedder_dict['wh50'] = (KeyedVectors.load("wordVectors/wh_w2v50.kv", mmap='r'), 50)
    word_embedder_dict['wh100'] = (KeyedVectors.load("wordVectors/wh_w2v100.kv", mmap='r'), 100)
    sif_w2v = SIF(word_embedder_dict['w2v50'][0], workers=2)
    sif_wh = SIF(word_embedder_dict['wh50'][0], workers=2)
    sif_w2v.train(splits)
    word_embedder_dict['SIF'] = (sif_w2v, 50)
    sif_wh.train(splits)
    word_embedder_dict['SIF_wh'] = (sif_wh, 50)
    return word_embedder_dict

freq_dict, vocab, all_sentences, split_sentences = wordFreq()
word_embedder_dict = load_word_embedder_dict(IndexedList(split_sentences))


def get_fse_embedding(instruct, embedderStr): 
    assert embedderStr in ['SIF', 'SIF_wh'], "embedderStr must be a pretrained fse embedding"
    embedder = word_embedder_dict[embedderStr][0]
    if instruct in all_sentences: 
        index = IndexedList(split_sentences).items.index(instruct.split())
        embedded = embedder.sv[index]
    else: 
        tmp = (instruct.split(), 0)
        embedded = embedder.infer([tmp])
    return embedded

def toNumerals(tokenizer_type, instructions): 
    ins_temp = []
    for instruct in instructions:
        embedding = torch.ones((PAD_LEN, 1))
        if tokenizer_type is 'BERT': 
            tokenized = torch.Tensor(bertTokenizer.encode(instruct)).unsqueeze(1)
        if tokenizer_type is 'gpt': 
            tokenized = torch.Tensor(gptTokenizer.encode(instruct)).unsqueeze(1)
        embedding[:tokenized.shape[0]] = tokenized
        ins_temp.append(embedding)
    return torch.stack(ins_temp).squeeze().long().to(device)


def get_batch(batch_size, tokenizer, task_type = None, instruct_mode = None):
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
        if instruct_mode == 'single': 
            batch = [instruct]*batch_size
            batch_target_index = batch_target_index * batch_size
            break
        batch.append(instruct)
        if instruct_mode == 'blocked': 
            batch = batch * batch_size
            batch_target_index = batch_target_index * batch_size
            break
    if tokenizer is not None: 
        batch = toNumerals(tokenizer, batch)
    return batch, batch_target_index

class Pretrained_Embedder(nn.Module): 
    def __init__(self, embedderStr):
        super(Pretrained_Embedder, self).__init__()
        self.embedderStr = embedderStr
        if embedderStr in ['w2v50', 'w2v100', 'wh50', 'wh100']:
            self.out_dim = word_embedder_dict[embedderStr][1]
        elif self.embedderStr in ['SIF', 'SIF_wh']: 
            self.out_dim = 50

    def forward(self, instruct): 
        embedded = []
        for i in range(len(instruct)):
            if self.embedderStr in ['w2v50', 'w2v100', 'wh50', 'wh100']:
                w2v = word_embedder_dict[self.embedderStr][0]
                instruct_list = []
                instruction = instruct[i].split()
                for i in range(PAD_LEN): 
                    #SHOULD USE PADDED SEQUENCE HERE
                    if i < len(instruction):
                        vec = w2v[instruction[i]]
                        instruct_list.append(vec)
                    else: 
                        zeros = np.zeros_like(vec)
                        instruct_list.append(zeros)
                embedded.append(np.array(instruct_list))
            if self.embedderStr in ['SIF', 'SIF_wh']: 
                embedded.append(get_fse_embedding(instruct[i], self.embedderStr))
        return torch.Tensor(np.array(embedded)).to(device)

class LastLinear(nn.Module): 
    def __init__(self, langModel): 
        super(LastLinear, self).__init__()
        self.langModel = langModel
        self.linear = nn.Linear(self.langModel.out_dim, len(task_list))

    def forward(self, x): 
        x = self.langModel(x)
        out = self.linear(x)
        return out


class LangModule(): 
    def __init__(self, langModel, instruct_mode = None): 
        self.langModel = langModel
        self.embedderStr = langModel.embedderStr
        if self.embedderStr in ['trainable50', 'trainable100']: 
            self.tokenizer_type = 'wikiText'
        elif self.embedderStr is 'gpt': 
            self.tokenizer_type = 'gpt'
        elif self.embedderStr is 'BERT': 
            self.tokenizer_type = 'BERT'
        else: 
            self.tokenizer_type = None
        self.loss_list = []
        self.val_loss_list = []
        self.instruct_mode = instruct_mode
        self.model_classifier = LastLinear(self.langModel).to(device)
        self.shuffled = False
        self.classifier_criterion = nn.CrossEntropyLoss()

        self.filename = self.embedderStr + '_' + str(self.langModel.out_dim) + '.pt'

    def train_classifier(self, batch_len, num_batches, epochs, optim_method = 'adam', lr=0.001, weight_decay=0, shuffle = False, train_out_only = False):
        self.shuffled = shuffle
        if optim_method == 'adam': 
            opt = optim.Adam(self.model_classifier.parameters(), lr, weight_decay=weight_decay)
        if optim_method == 'SGD': 
            opt = optim.SGD(self.model_classifier.parameters(), lr, weight_decay=weight_decay)
        if train_out_only: 
            opt = optim.Adam(self.model_classifier.linear.parameters(), lr, weight_decay=weight_decay)

        best_val_loss = 1e5
        self.model_classifier.train()
        for i in range(epochs):
            for j in range(num_batches): 
                opt.zero_grad()
                ins_temp, targets = get_batch(batch_len, tokenizer = self.tokenizer_type, instruct_mode = self.instruct_mode)
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
                    torch.save(self.model_classifier.state_dict(), 'LanguageModels/'+self.filename)
        self.model_classifier.load_state_dict(torch.load(self.filename))

    def get_val_loss(self): 
        total_loss = 0
        num_sentences = 0
        self.model_classifier.eval()
        for i, task in enumerate(task_list): 
            instructions = test_instruct_dict[task]
            if self.tokenizer_type is not None: 
                instructions = toNumerals(self.tokenizer_type, instructions)
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
            if self.tokenizer_type is not None: 
                instructions = toNumerals(self.tokenizer_type, instructions)
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
        self.langModel.eval()
        task_indices = []
        rep_tensor = torch.Tensor().to(device)
        for i, task in enumerate(task_list):
            instructions = instruct_dict[task]
            if self.tokenizer_type is not None: 
                instructions = toNumerals(self.tokenizer_type, instructions)
            out_rep = self.langModel(instructions)
            task_indices += ([i]*len(instructions))
            rep_tensor = torch.cat((rep_tensor, out_rep), dim=0)
        return task_indices, rep_tensor

    def plot_embedding(self, embedding_type, dim=2, train_only=False):
        assert embedding_type in ['PCA', 'tSNE'], "entered invalid embedding_type: %r" %embedding_type
        assert dim in [2, 3], "embedding dimension must be 2 or 3"

        train_indices, train_rep = self._get_instruct_rep(train_instruct_dict)
        test_indices, test_rep = self._get_instruct_rep(test_instruct_dict)
        if test_rep.dim()>2: 
            test_rep = test_rep.squeeze()
        if embedding_type == 'PCA':
            embedded_train = PCA(n_components=dim).fit_transform(train_rep.cpu().detach().numpy())
            embedded_test = PCA(n_components=dim).fit_transform(test_rep.cpu().detach().numpy())
        elif embedding_type == 'tSNE': 
            embedded_train = TSNE(n_components=dim).fit_transform(train_rep.cpu().detach().numpy())
            embedded_test = TSNE(n_components=dim).fit_transform(test_rep.cpu().detach().numpy())

        cmap = matplotlib.cm.get_cmap('hsv')
        norm = matplotlib.colors.Normalize(vmin=0, vmax=12)

        color_train = norm(np.array(train_indices).astype(int))
        color_test = norm(np.array(test_indices).astype(int))
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
        digits = np.arange(len(task_list))
        Patches = [mpatches.Patch(color=cmap(norm(d)), label=task_list[d]) for d in digits]
        Patches.append(Line2D([0], [0], marker='X', color='w', label='test data', markerfacecolor='grey', markersize=10))
        Patches.append(Line2D([0], [0], marker='o', color='w', label='train data', markerfacecolor='grey', markersize=10))

        plt.legend(handles=Patches)
        plt.show()

def plot_lang_perf(mod_dict, mode, smoothing): 
    if mode == 'train': 
        for label, mod in mod_dict.items(): 
            loss = smoothed_perf = gaussian_filter1d(mod.loss_list, sigma=smoothing)
            plt.plot(loss, label = label)
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


class gpt2(nn.Module): 
    def __init__(self, out_dim, d_reduce='avg'): 
        super(gpt2, self).__init__()
        from transformers import GPT2Model
        self.d_reduce = d_reduce
        self.embedderStr = 'gpt'
        self.out_dim = out_dim

        if self.d_reduce == 'linear':
            self.proj_out = nn.Sequential(nn.Linear(PAD_LEN*768, 768), nn.ReLU(), nn.Linear(768, self.out_dim), nn.ReLU())
        else: 
            self.proj_out = nn.Sequential(nn.Linear(768, self.out_dim), nn.ReLU())

        self.transformer = GPT2Model.from_pretrained('gpt2')
    def forward(self, x): 
        trans_out = self.transformer(x)[0]
        if self.d_reduce == 'linear': 
            out = self.proj_out(trans_out.flatten(1))
        elif self.d_reduce ==  'max': 
            trans_out = torch.max((trans_out), dim=1)
            out =self.proj_out(trans_out)
        elif self.d_reduce == 'avg': 
            trans_out = torch.mean((trans_out), dim =1) 
            out =self.proj_out(trans_out)
        return out

class BERT(nn.Module):
    def __init__(self, out_dim, d_reduce='avg'): 
        super(BERT, self).__init__()
        from transformers import BertModel
        self.d_reduce = d_reduce
        self.embedderStr = 'BERT'
        self.out_dim = out_dim

        if self.d_reduce == 'linear':
            self.proj_out = nn.Sequential(nn.Linear(PAD_LEN*768, 768), nn.ReLU(), nn.Linear(768, self.out_dim), nn.ReLU())
        else: 
            self.proj_out = nn.Sequential(nn.Linear(768, self.out_dim), nn.ReLU())

        self.transformer = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, x): 
        trans_out = self.transformer(x)[0]
        if self.d_reduce == 'linear': 
            out = self.proj_out(trans_out.flatten(1))
        elif self.d_reduce ==  'max': 
            trans_out = torch.max((trans_out), dim=1)
            out =self.proj_out(trans_out)
        elif self.d_reduce == 'avg': 
            trans_out = torch.mean((trans_out), dim =1) 
            out =self.proj_out(trans_out)
        elif self.d_reduce == 'avg_conv': 
            trans_out = nn.AvgPool2d((1, ), (2,1))
        return out


class SBERT(nn.Module): 
    def __init__(self, out_dim): 
        super(SBERT, self).__init__()
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')
        self.embedderStr = 'SBERT'
        self.out_dim = out_dim
        self.lin = nn.Sequential(nn.Linear(768, self.out_dim), nn.ReLU())
    def forward(self, x): 
        sent_embedding = np.array(self.model.encode(x))
        sent_embedding = self.lin(torch.Tensor(sent_embedding).to(device))
        return sent_embedding
        
class SIFmodel(nn.Module): 
    def __init__(self, SIF_type): 
        super(SIFmodel, self).__init__()
        self.out_shape = ['batch_len', 50]
        self.in_shape = ['batch_len', 50]
        self.out_dim =50 
        self.embedderStr = SIF_type
        self.identity = nn.Identity()

    def forward(self, x): 
        return torch.Tensor(self.identity(x))


class BoW(nn.Module): 
    def __init__(self): 
        super(BoW, self).__init__()
        self.embedderStr = 'BoW'
        self.out_dim = len(vocab)    

    def forward(self, x): 
        batch_vectorized = []
        for instruct in x: 
            out_vec = torch.zeros(self.out_dim)
            for word in instruct.split():
                index = vocab.index(word)
                out_vec[index] += 1
            batch_vectorized.append(out_vec)
        return torch.stack(batch_vectorized).to(device)
