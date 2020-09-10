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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.ndimage.filters import gaussian_filter1d

import pickle

import os
os.chdir('/home/reidar/Projects/LanguageCog/CogRNN')
from batchTaskedit import Task
task_list = Task.TASK_LIST

########SET UP SPECIFIC INSTRUCTIONS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_LEN = 20

import torchtext
from torchtext.data.utils import get_tokenizer
TEXT = torchtext.data.Field(fix_length = 15, tokenize=get_tokenizer("basic_english"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt, val_txt, test_txt)
ntokens = len(TEXT.vocab.stoi)

from transformers import GPT2Tokenizer, BertTokenizer
gptTokenizer = GPT2Tokenizer.from_pretrained('gpt2')
bertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# import pickle
# pickle.dump(rich_instruct_dict, open('rich_instruct_dict', 'wb'))
#pickle.dump(test_instruction_dict, open('test_instruction_dict', 'wb'))

rich_instruct_dict = pickle.load(open('rich_instruct_dict', 'rb'))
test_instruction_dict = pickle.load(open('test_instruction_dict', 'rb'))

swapped_task_list = ['Anti DM', 'Anti MultiDM', 'Anti Go', 'DMS', 'DNMC', 'Go', 'MultiDM', 'RT Go', 'DNMS', 'DMC', 'DM', 'Anti RT Go']
instruct_swap_dict = dict(zip(swapped_task_list, rich_instruct_dict.values()))

shuffled_rich_instruct_dict = pickle.load(open('shuffled_rich_instruct_dict', 'rb'))
shuffled_test_instruction_dict = pickle.load(open('shuffled_test_instruction_dict', 'rb'))


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
        instructions = rich_instruct_dict[task_type] + test_instruction_dict[task_type]
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

    return freq_dict, list(set(all_words)), all_sentences, split_sentences

test_instruction_dict['DMS']

def load_word_embedder_dict(): 
    word_embedder_dict = {}
    _,_,_, split_sentences = wordFreq()
    word_embedder_dict['w2v50'] = (KeyedVectors.load("w2v50.kv", mmap='r'), 50)
    word_embedder_dict['w2v100'] = (KeyedVectors.load("w2v100.kv", mmap='r'), 100)
    word_embedder_dict['wh50'] = (KeyedVectors.load("wh_w2v50.kv", mmap='r'), 50)
    word_embedder_dict['wh100'] = (KeyedVectors.load("wh_w2v100.kv", mmap='r'), 100)
    sif_w2v = SIF(word_embedder_dict['w2v50'][0], workers=2)
    sif_wh = SIF(word_embedder_dict['wh50'][0], workers=2)
    sif_w2v.train(s)
    word_embedder_dict['SIF'] = (sif_w2v, 50)
    sif_wh.train(s)
    word_embedder_dict['SIF_wh'] = (sif_wh, 50)
    gpt_hid = pickle.load(open('gpt_hid_dict', 'rb'))
    word_embedder_dict['gpt'] = gpt_hid
    return word_embedder_dict


freq_dict, vocab, all_sentences, split_sentences = wordFreq()
s = IndexedList(split_sentences)
word_embedder_dict = load_word_embedder_dict()
rand_embedder = np.random.randn(len(vocab), 50)

def get_fse_embedding(instruct, embedderStr): 
    assert embedderStr in ['SIF', 'SIF_wh'], "embedderStr must be a pretrained fse embedding"
    embedder = word_embedder_dict[embedderStr][0]
    if instruct in all_sentences: 
        index = s.items.index(instruct.split())
        embedded = embedder.sv[index]
    else: 
        tmp = (instruct.split(), 0)
        embedded = embedder.infer([tmp])
    return embedded

def get_rand_wordVec(instruct, embed_dim=50): 
    instruct_list = []
    instruction = instruct.split()
    for i in range(PAD_LEN): 
        #SHOULD USE PADDED SEQUENCE HERE
        if i < len(instruction):
            vec = np.matmul(np.eye(len(vocab))[vocab.index(instruction[i])], rand_embedder)
            instruct_list.append(vec)
        else: 
            zeros = np.zeros_like(vec)
            instruct_list.append(zeros)
    embedded = np.array(instruct_list)
    return embedded

def toNumerals(tokenizer_type, instructions): 
    ins_temp = []
    for instruct in instructions:
        embedding = torch.ones((PAD_LEN, 1))
        if tokenizer_type is 'BERT': 
            tokenized = torch.Tensor(bertTokenizer.encode(instruct)).unsqueeze(1)
        if tokenizer_type is 'gpt': 
            tokenized = torch.Tensor(gptTokenizer.encode(instruct)).unsqueeze(1)
        if tokenizer_type is 'wikiText': 
            tokenized = TEXT.numericalize([instruct.split()])
        embedding[:tokenized.shape[0]] = tokenized
        ins_temp.append(embedding)
    return torch.stack(ins_temp).squeeze().long().to(device)


def get_batch(batch_size, holdout_task, tokenizer, task_type = None, instruct_mode = None):
    batch = []
    batch_target_index = []
    for i in range(batch_size):
        task = task_type
        if task is None: 
            task = np.random.choice(task_list)
        if instruct_mode == 'instruct_swap': 
            instruct_dict = instruct_swap_dict
        else: 
            instruct_dict = rich_instruct_dict
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
            if self.embedderStr is 'rand': 
                embedded.append(get_rand_wordVec(instruct[i]))
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

        if instruct_mode == 'comb': 
            self.classifier_criterion = nn.BCEWithLogitsLoss()
        else:
            self.classifier_criterion = nn.CrossEntropyLoss()

        self.filename = self.embedderStr + '_' + str(self.langModel.out_dim) + '.pt'

    def train_classifier(self, batch_len, num_batches, epochs, optim_method = 'adam', lr=0.001, weight_decay=0, shuffle = False, 
                                            holdout_task = None, train_out_only = False):
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
                ins_temp, targets = get_batch(batch_len, holdout_task = holdout_task, tokenizer = self.tokenizer_type, instruct_mode = self.instruct_mode)
                tensor_targets = torch.Tensor(targets).to(device)
                out = self.model_classifier(ins_temp)
                if self.instruct_mode == 'comb': 
                    loss = self.classifier_criterion(out, tensor_targets.float())
                else: 
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
                    torch.save(self.model_classifier.state_dict(), self.filename)
        self.model_classifier.load_state_dict(torch.load(self.filename))

    def get_val_loss(self): 
        total_loss = 0
        num_sentences = 0
        self.model_classifier.eval()
        for i, task in enumerate(task_list): 
            instructions = test_instruction_dict[task]
            if self.tokenizer_type is not None: 
                instructions = toNumerals(self.tokenizer_type, instructions)
            out = self.model_classifier(instructions)
            if self.instruct_mode == 'comb': 
                tensor_targets = torch.Tensor(Task.COMB_VEC_DICT[task]).repeat(4, 1).to(device)
                loss = self.classifier_criterion(out, tensor_targets.float())
            else: 
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
            instructions = test_instruction_dict[task]
            if self.tokenizer_type is not None: 
                instructions = toNumerals(self.tokenizer_type, instructions)
            model_cat = torch.argmax(self.model_classifier(instructions), dim=1).cpu().numpy()
            for j in model_cat: 
                confuse_mat[i, j] += 1
                if j == i:
                    num_correct += 1  

            for j, sentence in enumerate(test_instruction_dict[task]):
                if self.instruct_mode == 'comb': 
                    model_out = self.model_classifier([sentence]).squeeze()
                    model_hot_encoding = np.round(nn.Sigmoid()(model_out).cpu().detach().numpy())
                    model_cat = None
                    for k, encoding in enumerate(list(Task.COMB_VEC_DICT.values())): 
                        if np.all(encoding == model_hot_encoding):
                            model_cat = k
                            confuse_mat[i, model_cat] += 1
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

    def plot_embedding(self, embedding_type):
        assert embedding_type in ['PCA', 'tSNE'], "entered invalid embedding_type: %r" %embedding_type

        train_indices, train_rep = self._get_instruct_rep(rich_instruct_dict)
        test_indices, test_rep = self._get_instruct_rep(test_instruction_dict)
        if test_rep.dim()>2: 
            test_rep = test_rep.squeeze()
        if embedding_type == 'PCA':
            embedded_train = PCA(n_components=2).fit_transform(train_rep.cpu().detach().numpy())
            embedded_test = PCA(n_components=2).fit_transform(test_rep.cpu().detach().numpy())
        elif embedding_type == 'tSNE': 
            embedded_train = TSNE(n_components=2).fit_transform(train_rep.cpu().detach().numpy())
            embedded_test = TSNE(n_components=2).fit_transform(test_rep.cpu().detach().numpy())

        fig, ax = plt.subplots(figsize=(12, 10))
        cmap = matplotlib.cm.get_cmap('hsv')
        norm = matplotlib.colors.Normalize(vmin=0, vmax=13)

        color_train = norm(np.array(train_indices).astype(int))
        color_test = norm(np.array(test_indices).astype(int))

        plt.scatter(embedded_train[:, 0], embedded_train[:, 1], c=cmap(color_train), cmap=cmap, s=100)
        plt.scatter(embedded_test[:, 0], embedded_test[:, 1], c=cmap(color_test), marker= "X", s=100)

        plt.setp(ax, xticks=[], yticks=[])
        plt.xlabel("PCA 1", fontsize = 18)
        plt.ylabel("PCA 2", fontsize = 18)

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


def max_pool(tensor_in): 
    max_pooled, _ = torch.max(tensor_in, dim=1)
    return max_pooled

def avg_pool(tensor_in): 
    return torch.mean(tensor_in, dim=1)

class gpt2(nn.Module): 
    def __init__(self, out_dim, pooling): 
        super(gpt2, self).__init__()
        from transformers import GPT2Model
        self.pooling = pooling
        self.embedderStr = 'gpt'
        self.out_dim = out_dim

        if pooling == 'lin_out':
            self.proj_out = nn.Sequential(nn.Linear(PAD_LEN*768, 768), nn.ReLU(), nn.Linear(768, self.out_dim), nn.ReLU())
        else: 
            self.proj_out = nn.Sequential(nn.Linear(768, self.out_dim), nn.ReLU())

        self.transformer = GPT2Model.from_pretrained('gpt2')
    def forward(self, x): 
        trans_out = self.transformer(x)[0]
        if self.pooling == 'lin_out': 
            out = self.proj_out(trans_out.flatten(1))
        elif self.pooling ==  'max': 
            trans_out = max_pool(trans_out)
            out =self.proj_out(trans_out)
        elif self.pooling == 'avg': 
            trans_out = avg_pool(trans_out)
            out =self.proj_out(trans_out)
        return out

class BERT(nn.Module):
    def __init__(self, out_dim, pooling): 
        super(BERT, self).__init__()
        from transformers import BertModel
        self.pooling = pooling
        self.embedderStr = 'BERT'
        self.out_dim = out_dim

        if pooling == 'lin_out':
            self.proj_out = nn.Sequential(nn.Linear(PAD_LEN*768, 768), nn.ReLU(), nn.Linear(768, self.out_dim), nn.ReLU())
        else: 
            self.proj_out = nn.Sequential(nn.Linear(768, self.out_dim), nn.ReLU())

        self.transformer = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, x): 
        trans_out = self.transformer(x)[0]
        if self.pooling == 'lin_out': 
            out = self.proj_out(trans_out.flatten(1))
        elif self.pooling ==  'max': 
            trans_out = max_pool(trans_out)
            out =self.proj_out(trans_out)
        elif self.pooling == 'avg': 
            trans_out = avg_pool(trans_out)
            out =self.proj_out(trans_out)
        return out

class SBERT(nn.Module): 
    def __init__(self, out_dim, d_reduce): 
        super(SBERT, self).__init__()
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')
        self.embedderStr = 'SBERT'
        self.out_dim = out_dim
        self.d_reduce = d_reduce 
        if d_reduce == 'lin_out': 
            self.lin = nn.Sequential(nn.Linear(768, self.out_dim), nn.ReLU())
    def forward(self, x): 
        sent_embedding = np.array(self.model.encode(x))
        if self.d_reduce == 'PCA': 
            sent_embedding = np.array(self.model.encode(x))
            sent_embedding = torch.Tensor(PCA(n_components=self.out_dim).fit_transform(sent_embedding))
        if self.d_reduce == 'lin_out': 
            sent_embedding = self.lin(torch.Tensor(sent_embedding).to(device))
        return sent_embedding
        

class SIFmodel(nn.Module): 
    def __init__(self, SIF_type): 
        super(SIFmodel, self).__init__()
        self.embed_dim = 50 
        self.out_shape = ['batch_len', 50]
        self.in_shape = ['batch_len', 50]
        self.out_dim =50 
        self.embedderStr = SIF_type
        self.identity = nn.Identity()

    def forward(self, x): 
        return torch.Tensor(self.identity(x))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class LangTransform(nn.Module): 
    def __init__(self, embedderStr, out_dim, d_model, num_layers, nheads, dim_ff, drop_p=0.1, usePretrain = False): 
        super(LangTransform, self).__init__()
        assert embedderStr in ['w2v50', 'w2v100', 'wh50', 'wh100', 'rand', 'trainable50', 'trainable100'],'embedderSTR must take a word embedding'

        self.embedderStr = embedderStr
        if embedderStr not in ['trainable50', 'trainable100']: 
            self.encoder = Pretrained_Embedder(self.embedderStr)
            self.embed_dim = self.encoder.embed_dim
        elif embedderStr is 'trainable50': 
            self.encoder = nn.Embedding(len(TEXT.vocab.stoi), 50)
            self.embed_dim = 50
        elif embedderStr is 'trainable100': 
            self.encoder = nn.Embedding(len(TEXT.vocab.stoi), 100)
            self.embed_dim = 100

        self.out_dim = out_dim
        self.in_shape = ['batch_len', PAD_LEN, self.embed_dim]
        self.out_shape = ['batch_len', self.out_dim]

        self.drop_p = drop_p
        self.num_layers = num_layers
        self.dim_ff = dim_ff
        self.d_model = d_model
        self.nheads = nheads
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(self.d_model, dropout=self.drop_p)
        encoder_layers = TransformerEncoderLayer(self.d_model, self.nheads, self.dim_ff, dropout=self.drop_p)
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.num_layers)
        self.proj_out = nn.Sequential(nn.Linear(PAD_LEN*self.d_model, self.out_dim), nn.ReLU())

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.proj_out(output.flatten(1))
        return output

class RecurrentWord2Task(nn.Module): 
    def __init__(self, embedderStr, hid_dim, out_dim, drop_p = 0.1): 
        super(RecurrentWord2Task, self).__init__()
        assert embedderStr in ['w2v50', 'w2v100', 'wh50', 'wh100', 'rand'],'embedderSTR must take a word embedding'
        self.embedderStr = embedderStr
        self.embed_dim = word_embedder_dict[embedderStr][1]
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        self.in_shape = ['batch_len', PAD_LEN, self.embed_dim]
        self.out_shape = ['batch_len', self.out_dim]

        self.rnn = nn.GRU(self.embed_dim, self.hid_dim, batch_first=True)
        self.lin_out = nn.Linear(self.hid_dim, self.out_dim)
        self.word_drop = nn.Dropout(drop_p)
    def forward(self, x): 
        x = self.word_drop(x)
        _, h_T = self.rnn(x)
        out = self.lin_out(h_T)
        return out
    
    def initHidden(self):
        return torch.full((1, 1, self.hid_dim))

