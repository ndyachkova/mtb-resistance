import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import time
from gensim.models import Word2Vec
from tqdm import tqdm
from sklearn.metrics import roc_auc_score as auc_score
# roc_auc_score(y_true, y_pred), y_pred is probability of the greater class
from sklearn.metrics import f1_score, recall_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

#torch.manual_seed(1)

data_path = '../data'

total = pd.read_csv('../data/annotation.csv')
start = dict(zip(total['gene'], total['start']))
end = dict(zip(total['name'], total['end']))
strand = dict(zip(total['name'], total['strand']))


def make_feature_list(file, samples, verbose=False, filter_rare=False, thr=5, thr_set=[], filter_possible=False, possible=[]):
    """
    list of lists; each sublist - features of a sample.

    file: string, directory of files with features
    samples: list, list of sample IDs to use

    thr: int, minimum frequency
    thr_set: list of samples to use to calculate frequencies.

    filter_possible: bool, whether to only keep a given subset of features.
    possible: set of featurse to keep.

    returns:
    features: list of lists
    """
    features = [0] * len(samples)
    # dict for frequencies
    fts = dict()
    # if not set to calculate thresholds is given, use all samples
    if not len(thr_set):
        thr_set = samples
    # reading
    for i in range(len(samples)):
        # each file is a single line of space-separated features
        with open(file + samples[i] + '_result.tsv', 'r') as f:
            lines = f.readlines()
        if len(lines) > 1:
            print('length error:', len(lines), 'in', samples[i])
        features[i] = lines[0].split()
        # select features in allowed subset
        if filter_possible:
            features[i] = [x for x in features[i] if x in possible]
        # count feature frequency
        if filter_rare:
            if samples[i] in thr_set:
                for ft in features[i]:
                    fts[ft] = fts.get(ft, 0) + 1
    #names = [line.split()[0] for line in lines if line.split()[0] in samples]
    # filter rare features
    if filter_rare:
        if verbose:
            print('Frequency threshold:', thr)
        for i in range(len(features)):
            if verbose:
                print(samples[i], len(features[i]), 'features', end='\t')
            features[i] = [ft for ft in features[i] if fts.get(ft, 0) >= thr and fts.get(ft, 0) < len(thr_set) - thr]
            if verbose:
                print('after filtration', len(features[i]), 'features')
    return features


def sort_key(ft):
    """
    Key function for sorting along the genome.
    """
    # if agregated: start of gene
    if ft.find('_PF') != -1:
        if strand[ft[:ft.find('_PF')]] == '+':
            return start[ft[:ft.find('_PF')]]
        else:
            return end[ft[:ft.find('_PF')]]
    # if broken
    if ft.find('broken_gene') != -1 or ft.find('broken_start') != -1:
        if strand[ft[:ft.find('#')]] == '+':
            return start[ft[:ft.find('#')]]
        else:
            return end[ft[:ft.find('#')]]
    # broken end
    if ft.find('broken_end') != -1:
        if strand[ft[:ft.find('#')]] == '+':
            return end[ft[:ft.find('#')]]
        else:
            return start[ft[:ft.find('#')]]
    # alt start
    if ft.find('alternative_start') != -1:
        if strand[ft[:ft.find('#')]] == '+':
            return start[ft[:ft.find('#')]]
        else:
            return end[ft[:ft.find('#')]]
    ft = ft.split('#')
    # if only coordinate (no gene)
    if ft[0] == '-':
        return int(ft[1]) # feature is -, coord, description
    
    # if gene + coord: gene start + coord
    try:
        t = int(ft[1])
    except:
        print(ft)
    if strand[ft[0]] == '+':
        return start[ft[0]] + int(ft[1])
    else:
        return end[ft[0]] - int(ft[1])


# functions for loss adjustment
class loss_function(nn.Module):
    def __init__(self, loss_fn, f:callable, coef=1):
        super(loss_function, self).__init__()
        self.coef = coef
        self.loss_fn = loss_fn
        self.f = f

    def forward(self, pred, true, attn, unmatched):
        # Compute the loss
        loss = [0] * true.shape[-1]
        for i in range(len(loss)):
            loss[i] = self.loss_fn[i](pred[:, i], true[:, i])
            #loss[i] = self.loss_fn[i](pred[:, i], true[:, i])  - self.coef * torch.sum(attn[0] * torch.log(attn[0] / attn[1]))
        #penalty = self.coef * self.f(unmatch(attn, unmatched))
        #print('loss:', loss, 'pred:', pred.shape)
        penalty = []
        for d2 in range(len(attn[0])):
            for d1 in range(d2):
                penalty.append(self.coef * self.f(unmatch(attn, unmatched, d1, d2)))
        #penalty = self.coef * torch.mean(torch.stack([torch.sum(att[0] * torch.log(att[0] / att[1])) for att in attn]))
        l = sum(loss) / len(loss) - sum(penalty) / (len(penalty))
        # this only works the once
        #if l.requires_grad == False:
        #    l.requires_grad = True
        return l, loss, penalty
        #return torch.sum(attn[0] * torch.log(attn[0] / attn[1]))


def unmatch(attn, unmatched, d1, d2):
    """
    Match features in attention tensors by removing unmatched features.
    d1, d2: int, indeces of drugs for which to build attention vectors.
    """
    # attn: list of list of tensors
    # samples
    attn_new = [0] * len(attn)
    # if one sample: unmatches is list for drugs, for each drug - indeces to be skipped
    # iterate over samples
    for i in range(len(attn)):
        # for each drug
        attn_new[i] = [0, 0]
        if i in unmatched and (d1, d2) in unmatched[i]:
            #print(i, [len(v) for v in attn[i]], unmatched[i])
            # iterate over drugs
            for j in range(2):
                d = [d1, d2][j]
                tmp = np.ones(len(attn[i][d]), dtype=bool)
                tmp[unmatched[i][(d1, d2)][j]] = False
                attn_new[i][j] = attn[i][d][tmp]
            attn_new[i] = torch.stack(attn_new[i])
        else:
            attn_new[i] = torch.stack([attn[i][d1], attn[i][d2]])
    return attn_new


def kl(attn):
    """
    Kullback–Leibler divergence with all features. Assumes length of attn equals 2.
    DOES NOT WORK FOR SEVERAL SAMPLES.
    """
    return torch.sum(attn[0] * torch.log(attn[0] / attn[1]))

def kl_multi(attn):
    """
    Kullback–Leibler divergence with all features. Assumes length of attn equals 2.
    Works even if attn is a list of attention vectors for several samples.
    """
    return torch.mean(torch.stack([torch.sum(att[0] * torch.log(att[0] / att[1])) for att in attn]))


def js(attn):
    """
    Jensen–Shannon divergence with all features. Assumes length of attn equals 2.
    DOES NOT WORK FOR SEVERAL SAMPLES.
    """
    return 0.5 * (kl([attn[0], 0.5*(attn[0] + attn[1])]) + kl([attn[1], 0.5*(attn[0] + attn[1])]))


def js_multi(attn):
    """
    Jensen–Shannon divergence with all features.
    """
    return torch.mean(torch.stack([0.5 * (kl([att[0], 0.5*(att[0] + att[1])]) + kl([att[1], 0.5*(att[0] + att[1])])) for att in attn]))


def fold_split(drugs, fld, include_neither=False, validation=False, seed=29051453, verbose=False, table_only=False):
    """
    Return train and test samples.
    include_neither: for w2v: consider samples w/o target for any drugs
    """

    # open file with info on resistance
    data = pd.read_csv('../data/all_resistance.csv')
    # selecting columns for chosen drugs
    data = data[['id'] + drugs].copy()

    # removing nan
    data_na = data.dropna().copy()

    # folds
    step = int(data_na.shape[0] / 5)
    r = data_na.shape[0] % 5

    rng = np.random.default_rng(seed=seed)
    data_na['fld'] = rng.choice(['1']*step +['2']*step +['3']*step +['4']*step +['5']*(step+r), data_na.shape[0], replace=False)

    if include_neither:
        # selecting ids w/o info for any of the drugs
        # bc those with info on one can be used as validation set
        tmp = data.fillna(-1)
        tmp = tmp[tmp[drugs].sum(axis=1) == -1 * len(drugs)].copy()
        # we need to include ids w/o info in the w2v train set
        # so their fold needs to not be the current fold
        tmp['fld'] = '1' if fld != '1' else '2'

        # ids with info on all or none
        data = pd.concat((data_na, tmp)).copy()
    else:
        if validation:
            # data that is not included in train or test
            data_valid = data.loc[~data['id'].isin(data_na['id'])]
            valid = [0] * len(drugs)
            # samples with info for each drug
            for i in range(len(drugs)):
                tmp = data_valid[['id', drugs[i]]].dropna()
                valid[i] = (list(tmp['id']), tmp[[drugs[i]]].to_numpy(dtype=int))
        data = data_na
        # resistance info: an array for each sample
        y_train = data.loc[data['fld'] != fld][drugs].to_numpy(dtype=int)
        y_test =  data.loc[data['fld'] == fld][drugs].to_numpy(dtype=int)
    # only get the table with id + resistance + fld
    if table_only:
        return data
    # test set
    test_samples = list(data.loc[data['fld'] == fld]['id'])
    train_samples = list(data.loc[data['fld'] != fld]['id'])
    if include_neither:
        return train_samples, test_samples
    else:
        if validation:
            return train_samples, test_samples, y_train, y_test, valid
        else:
            return train_samples, test_samples, y_train, y_test


# updated for multi
def add_agr(features, samples_IDs, drugs, fld, agr_name, use_unmatched=False, path='../data/agr_features/', split_by_drug=False, verbose=False, thr=0, thr_set=[], filter_possible=False, possible=[], verbose_debug=False):
    """
    Add aggregated features.

    paramteters:
    features: list of lists of string, given features for each sample
    sample_IDs: list of strings, list of IDs of samples
    drugs: list of strings, drugs
    fld: string, fold
    agr_name: name of file with aggregated features to add to pre-determined path
    split_by_drug: bool, default:False, create separate lists of features for each drug

    thr: int, minimum allowed frequency of a feature
    thr_set: list of samples to use to calculate feature frequency.

    filter_possible: bool, whether to only keep a given subset of features.
    possible: set of featurse to keep.
    """
    # agr_name: {drug}{agr_name}{fld}.txt
    with open(f'{path}{"_".join([x[:3] for x in drugs])}{agr_name}{fld}.txt') as f:
        lines = f.readlines()
    # structure of file: each line is ID then space-separated features

    # list of IDs
    agr = [line.split()[0] for line in lines]
    # filter the features by frequency if necessary
    freqs = dict()
    if thr > 0:
        if verbose:
            print('Frequency threshold:', thr)
        for line in lines:
            if line.split()[0] in thr_set:
                for l in line.split()[1:]:
                    freqs[l] = freqs.get(l, 0) + 1
        for i in range(len(lines)):
            # check if this is info for a sample that we need
            if agr[i] in samples_IDs:
                smpl = agr[i]
                if verbose:
                    print(smpl, len(lines[i].split())-1, ' aggregated features', end=' ')
                lines[i] = [ft for ft in lines[i].split()[1:] if freqs.get(ft, 0) >= thr and freqs.get(ft, 0) < len(thr_set) - thr]
                if verbose:
                    print('after filtration', len(lines[i]), 'features', end=' ')
                # select features in allowed subset
                if filter_possible:
                    lines[i] = [x for x in lines[i] if x in possible]
                    if verbose:
                        print('after filtration of possible', len(lines[i]), 'features')
                else:
                    if verbose:
                        print()
    else:
        for i in range(len(lines)):
            if agr[i] in samples_IDs:
                lines[i] = lines[i].split()[1:]
                smpl = agr[i]
                if verbose:
                    print(smpl, len(lines[i]), ' aggregated features', end=' ')
                # select features in allowed subset
                if filter_possible:
                    lines[i] = [x for x in lines[i] if x in possible]
                    if verbose:
                        print('after filtration of possible', len(lines[i]), 'features')
                else:
                    if verbose:
                        print()
    
    
    if split_by_drug:
        if use_unmatched:
            res = [0] * len(features)
            fts = [0] * len(drugs)
            unmatched = dict()
            for i in range(len(features)):
                # samples for this drug
                res[i] = [0] * len(drugs)
                # iterate over samples
            for i in range(len(features)):
                #if i == 1476:
                #    verbose = True
                #else:
                #    verbose = False
                unmatched[i] = dict()
                for d in range(len(drugs)):
                    # if there are agr features for this sample, add them
                    if samples_IDs[i] in agr:
                        # agr features for this drug have its first three letters in the name
                        fts[d] = [l for l in lines[agr.index(samples_IDs[i])] if l.find(drugs[d][:3]) != -1]
                        res[i][d] = features[i] + fts[d]
                        if verbose:
                            print(i, drugs[d], samples_IDs[i], sorted(fts[d]))
                    else:
                        print('Warning: no aggregated features for', samples_IDs[i])
                    # sort features
                    res[i][d].sort(key=sort_key)
                
                    # for each drug pair we need to marke unmatched agr features
                    # iterate over drugs that have already been processed
                    # (for which we have lists of domain features)
                    for d2 in range(d):
                        # features present in both drugs
                        # sometimes there is a dot in the gene name, so we need to make sure we find one in domain name
                        s = set([x[:x.find('.', x.find('PF'))] for x in fts[d]])
                        s = s & set([x[:x.find('.', x.find('PF'))] for x in fts[d2]])
                        if verbose:
                            print(drugs[d2], drugs[d], sorted(list(s)))
                        # unmatched features for both drugs
                        unm1 = [res[i][d2].index(x) for x in fts[d2] if x[:x.find('.', x.find('PF'))] not in s]
                        unm2 = [res[i][d].index(x) for x in fts[d] if x[:x.find('.', x.find('PF'))] not in s]
                        # if either drug has unmatched features, save them to dict
                        if unm1 or unm2:
                            unmatched[i][(d2, d)] = [unm1, unm2]
                            # check that after matching the lengths are the same
                            if len(res[i][d2]) - len(unmatched[i][(d2, d)][0]) != len(res[i][d]) - len(unmatched[i][(d2,d)][1]):
                                print('ERROR: lengths do not match after unmatching:', drugs[d2], drugs[d], i, samples_IDs[i])
                # if there are no unmatched features for any drug pair: we don't need an empty dict
                if unmatched[i] == dict():
                    unmatched.pop(i)
            return res, unmatched
        # for pre-train e.g. we don't need info for unmatched domain features.
        else:
            res = [0] * len(features)
            fts = [0] * len(drugs)
            for i in range(len(features)):
                # for each sample: len(drugs) lists - one for each drug
                res[i] = [0] * len(drugs)
                # iterate over samples
                for d in range(len(drugs)):
                    # if there are agr features for this sample, add them
                    if samples_IDs[i] in agr:
                        # agr features for this drug have its first three letters in the name
                        fts[d] = [l for l in lines[agr.index(samples_IDs[i])] if l.find(drugs[d][:3]) != -1]
                        res[i][d] = features[i] + fts[d]
                        if verbose_debug:
                            print(i, drugs[d], samples_IDs[i], sorted(fts[d]))
                    else:
                        print('Warning: no aggregated features for', samples_IDs[i])
                    # sort features
                    res[i][d].sort(key=sort_key)
            return res, {}
    else:
        for i in range(len(features)):
            # if there are agr features for this sample, add them
            if samples_IDs[i] in agr:
                features[i] += lines[agr.index(samples_IDs[i])]
                if verbose_debug:
                    print(samples_IDs[i], sorted(lines[agr.index(samples_IDs[i])]))
            else:
                print('Warning: no aggregated features for', samples_IDs[i])
            # sort features
            features[i].sort(key=sort_key)
        return features


# updated for multi
def prep(drugs, fld, feature_lists, agr_name, seed=29051453, verbose=False, thr=5):
    """
    Preparing the data set for Word2Vec training for given fold of given drugs.
    """        
    # list of all samples
    with open('../data/all_samples.txt') as f:
        all_samples = set(list(map(lambda x: x[:x.find('_')], f.readlines())))
    
    # get train-test split
    train_samples, test_samples = fold_split(drugs, fld, include_neither=True, seed=seed, verbose=verbose)

    # set for w2v: all except test
    train_w2v_IDs = list(all_samples - set(test_samples))
    train_w2v = make_feature_list(feature_lists, train_w2v_IDs, filter_rare=True, 
                                  thr=thr, thr_set=train_samples, verbose=verbose)
    
    ## add agr features
    train_w2v = add_agr(train_w2v, train_w2v_IDs, drugs, fld, agr_name, split_by_drug=False, verbose=verbose,
                        thr=thr, thr_set=train_samples)
    return train_w2v


# updated for multi
def train_func_w2v(train_w2v, drugs, fld, w2v_name='', sizes=[100], win=0, verbose=False):
    """
    Training Word2Vec model.
    """
    if w2v_name:
        if w2v_name[-1] != '_':
            w2v_name += '_'
    if win == 0:
        win = max([len(l) for l in train_w2v])
    if verbose:
        print('window size is', win)
    for sz in sizes:
        starttime = time.perf_counter()
        model = Word2Vec(sentences=train_w2v, vector_size=sz, window=win, min_count=1, workers=4, sg=0)
        duration = timedelta(seconds=time.perf_counter()-starttime)
        if verbose:
            print(sz, 'took: ', duration)
        model.save(f"word2vec_models/word2vec_{'_'.join([x[:3] for x in drugs])}_{fld}_{w2v_name}{sz}.model")


# updated for multi
def run_w2v(drugs, fld, feature_lists, agr_name, w2v_name='', sizes=[100], win=0, verbose=False, thr=5):
    train_w2v = prep(drugs, fld, feature_lists, agr_name, verbose=verbose, thr=thr)
    if verbose:
        print(drugs, 'fold', fld)
    train_func_w2v(train_w2v, drugs, fld, w2v_name=w2v_name, sizes=sizes, win=win, verbose=verbose)

    
# here we start lstm-related functions

# updated for multi
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        # lstm_output = [batch size, seq_len, hidden_dim]
        attention_scores = self.attn(lstm_output)
        # attention_scores = [batch size, seq_len, 1]
        #attention_scores = attention_scores.squeeze(-1)
        # attention_scores = [batch size, seq_len]
        return F.softmax(attention_scores, dim=-2)


class lstm_attn_multi(nn.Module):
    def __init__(self, drugs, fld, input_dim, hidden_dim, m_name='', bidirectional=False, num_layers=1, dropout=0, zero=False, mul=1):
        #(self, input_dim, hidden_dim, drugs, bidirectional=False, num_layers=1, dropout=0, zero=False, mul=1):
        super().__init__()
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        # input.size(-1) must be equal to input_size.
        self.drugs = drugs
        self.fld = fld
        self.zero = zero
        self.mul = mul
        self.m_name = m_name
        self.lstm = [0] * len(drugs)
        for i in range(len(drugs)):
            self.lstm[i] = nn.LSTM(input_dim, hidden_dim, bidirectional=bidirectional, batch_first=True, num_layers=num_layers, dropout=dropout)
        self.lstm = nn.ParameterList(self.lstm)
        # attention layer
        self.attn = [0] * len(drugs)
        for i in range(len(drugs)):
          self.attn[i] = Attention(hidden_dim + hidden_dim * bidirectional)
        self.attn = nn.ParameterList(self.attn)
        # The linear layer that maps from hidden state space to tag space
        self.linear = [0] * len(drugs)
        for i in range(len(drugs)):
          self.linear[i] = nn.Linear(hidden_dim + hidden_dim * bidirectional, 2)
        self.linear = nn.ParameterList(self.linear)

    def forward(self, data):
        # outputs all states, (hidden_state, cell_state)
        out = [0] * len(self.drugs)
        for i in range(len(self.drugs)):
            out[i], _ = self.lstm[i](data[i]) # mod for split_by_drug: data -> data[i]
        #print('out:', out.shape)
        # x = hidden[1]  # later update
        weights = [0] * len(self.drugs)
        for i in range(len(self.drugs)):
          weights[i] = self.attn[i](out[i])#.unsqueeze(-1)  # adding an extra dimention at the end
          #print(out.shape)
          #print(weights.shape)
          if self.zero:
            thr = 1/weights[i].shape[0] * self.mul
            weights[i] = weights[i].masked_fill(weights[i] < thr, 0)
        x = [0] * len(self.drugs)
        for i in range(len(self.drugs)):
          w = out[i] * weights[i]
          #print(w.shape)
          w = w.sum(dim=0)
          x[i] = self.linear[i](w)
        return torch.stack(x), weights


# updated for split_by_drug
def preproc(model, drugs, data, unmatched, match=False):
    """
    Use w2v model to convert data into vectors, skipping features for which encoding doesn't exist.
    
    model: Word2vec model
    drugs: list of strings, drugs
    data: list of lists, each sublist - featues for a sample, each sublist of that - list of strings (features) for each drug
    match: if all the features are expected to be present in model keys (is true for train, false for test).
    """
    possibles = set(model.wv.key_to_index.keys())
    # feature vectors
    tr = [0] * len(data)
    # feature names
    names = [0] * len(data)

    for i in range(len(data)):
        tr[i] = [0] * len(drugs)
        names[i] = [0] * len(drugs)

    for i in range(len(data)):
        for d in range(len(drugs)):
            names[i][d] = [x for x in data[i][d] if x in possibles]
            if len(names[i][d]) == 0:
                print('Error: no possible features for sample', i, 'for drug', drugs[d], 'out of', len(data[i][d]))
                tr[i][d] = [[]]
                continue
            tr[i][d] = model.wv[names[i][d]]
            # if some features were dropped, we need to adjust unmatched indeces
            if len(names[i][d]) < len(data[i][d]) and i in unmatched:
                # iterate over drugs
                for d2 in range(d):
                    tmp = np.zeros(len(data[i][d]))
                    # now mismatched for current drug are non-zeros
                    tmp[unmatched[i][(d2,d)][1]] = 1
                    # now non-zeros are mismatches in the new feature order
                    tmp = np.array([tmp[j].item() for j in range(len(data[i][d])) if data[i][d][j] in possibles])
                    # newly missing agr features that are now unmatched
                    # (need to do this before unmatched indeces are updated)
                    unm = [data[i][d][j] for j in range(len(data[i][d])) if data[i][d][j] not in possibles and data[i][d][j].find('_PF') != -1 and j not in unmatched[i][(d2, d)][1]]
                    # if they are present for the other drug, they need to be marked unmatched
                    for ft in unm:
                        # gene_domain portion of the feature name
                        ft_s = ft[:ft.find('.', ft.find('PF'))]
                        # if the drug was processed before, the index is from the new `names` list
                        unmatched[i][(d2,d)][0] += [j for j in range(len(names[i][d2])) if names[i][d2][j].find(ft_s) != -1]
                    unmatched[i][(d2, d)][1] = tmp.nonzero()[0].tolist()
                        
                for d2 in range(d+1, len(drugs)):
                    tmp = np.zeros(len(data[i][d]))
                    # now mismatched for current drug are non-zeros
                    tmp[unmatched[i][(d,d2)][0]] = 1
                    # now non-zeros are mismatches in the new feature order
                    tmp = np.array([tmp[j].item() for j in range(len(data[i][d])) if data[i][d][j] in possibles])
                    # newly missing agr features that are now unmatched
                    # (need to do this before unmatched indeces are updated)
                    unm = [data[i][d][j] for j in range(len(data[i][d])) if data[i][d][j] not in possibles and data[i][d][j].find('_PF') != -1 and j not in unmatched[i][(d, d2)][0]]
                    # if they are present for the other drug, they need to be marked unmatched
                    for ft in unm:
                        # gene_domain portion of the feature name
                        ft_s = ft[:ft.find('.', ft.find('PF'))]
                        # if the drug was processed before, the index is from the new `names` list
                        unmatched[i][(d,d2)][1] += [j for j in range(len(data[i][d2])) if data[i][d2][j].find(ft_s) != -1]
                    unmatched[i][(d, d2)][0] = tmp.nonzero()[0].tolist()
        if i in unmatched:
            for d1, d2 in unmatched[i]:
                if len(tr[i][d1]) - len(unmatched[i][(d1,d2)][0]) != len(tr[i][d2]) - len(unmatched[i][(d1,d2)][1]):
                    print('ERROR: lengths do not match after preprocessing:', drugs[d1], drugs[d2], i)
                    print(drugs[d1], [(j, data[i][d1][j]) for j in range(len(data[i][d1])) if data[i][d1][j] not in possibles])
                    print(drugs[d2], [(j, data[i][d2][j]) for j in range(len(data[i][d2])) if data[i][d2][j] not in possibles])
                        

    # check that all features have a corresponding key in model
    if match:
        for i in range(len(tr)):
            for j in range(len(tr[i])):
                if len(tr[i][j]) != len(data[i][j]):
                    print('Error: only', len(tr[i][j]), 'features out of', len(data[i][j]), 'for sample', i, 'drug', drugs[j])
                    break
    return tr, names, unmatched


# updated for multi
def prepare_given_data(drugs, samples, y, fld, w2v_name, feature_lists, agr_name, use_unmatched=False, return_names=False, seed=29051453, verbose=False):
    """
    Making list of features from given list of samples.

    use_unmatched, bool: calculate lists of unmatched features for each pairs. If False unmatched is {}.
    """

    # list of lists of features for each sample
    data = make_feature_list(feature_lists, samples)
    
    # add agr here
    data, unmatched = add_agr(data, samples, drugs, fld, agr_name, split_by_drug=True, use_unmatched=use_unmatched)
    
    # now data is:
    # list for samples
    # for each sample: list of features for each drug
    ######

    #model = Word2Vec.load(f"word2vec_models/word2vec_{drug}_{fld}_100.model")
    model = Word2Vec.load(f"word2vec_models/word2vec_{'_'.join([x[:3] for x in drugs])}_{fld}_{w2v_name}.model")

    # match: expect that all features in train set are in w2v model
    # may no longer be true if we filter w2v data and train data separately
    # feature may be acceptable for w2v but too rare in train
    vectors, names, unmatched = preproc(model, drugs, data, unmatched) # should add no_ft as 4th returned
        
    # for each drug train_vectors are a list of tensors
    vectors = [[torch.Tensor(x) for x in v] for v in vectors]
    y = [torch.Tensor(x) for x in y]
    y = torch.stack(y).long()

    if return_names:
        return names, vectors, y, unmatched
        #return train_samples, train_names, train_vectors, y_train, test_samples, test_names, test_vectors, y_test
        #return train_samples, train_vectors, y_train, test_samples, test_vectors, y_test
    return vectors, y, unmatched



# updated for multi
def prepare_data(drugs, fld, w2v_name, feature_lists, agr_name, use_unmatched=False, return_names=False, seed=29051453, verbose=False, load_saved=''):
    """
    Making train and test sets.

    load_saved: if not '', it is a path to pre-saved data.
    """
    if load_saved:
        train_vectors = torch.load(load_saved + 'train_vectors.pt', weights_only=True)
        y_train = torch.load(load_saved + 'y_train.pt', weights_only=True)
        test_vectors = torch.load(load_saved + 'test_vectors.pt', weights_only=True)
        y_test = torch.load(load_saved + 'y_test.pt', weights_only=True)

        if use_unmatched:
            with open(load_saved + 'unmatched_train.pkl', 'rb') as f:
                unmatched_train = pickle.load(f)

            with open(load_saved + 'unmatched_test.pkl', 'rb') as f:
                unmatched_test = pickle.load(f)
        else:
            unmatched_train = {}
            unmatched_test = {}
        
        with open(load_saved + 'train_names.pkl', 'rb') as f:
            train_names = pickle.load(f)

        with open(load_saved + 'test_names.pkl', 'rb') as f:
            test_names = pickle.load(f)
        
        if return_names:
            return train_names, train_vectors, y_train, test_names, test_vectors, y_test, unmatched_train, unmatched_test
            #return train_samples, train_names, train_vectors, y_train, test_samples, test_names, test_vectors, y_test
            #return train_samples, train_vectors, y_train, test_samples, test_vectors, y_test
        return train_vectors, y_train, test_vectors, y_test, unmatched_train, unmatched_test
    # get train-test split
    train_samples, test_samples, y_train, y_test = fold_split(drugs, fld, include_neither=False, 
    seed=seed, verbose=verbose)

    #model = Word2Vec.load(f"word2vec_models/word2vec_{drug}_{fld}_100.model")
    model = Word2Vec.load(f"word2vec_models/word2vec_{'_'.join([x[:3] for x in drugs])}_{fld}_{w2v_name}.model")
    possibles = set(model.wv.key_to_index.keys())

    # list of lists of features for each sample, only keep w2v-possible features
    test = make_feature_list(feature_lists, test_samples, filter_possible=True, possible=possibles, verbose=verbose)
    train = make_feature_list(feature_lists, train_samples, filter_possible=True, possible=possibles, verbose=verbose)
    # we do not need to filter rare here; w2v training data was filtered so w2v will filter it for us

    # add agr here, only keep w2v-possible options
    train, unmatched_train = add_agr(train, train_samples, drugs, fld, agr_name, split_by_drug=True, verbose=verbose,
                                        use_unmatched=use_unmatched, filter_possible=True, possible=possibles)
    test, unmatched_test = add_agr(test, test_samples, drugs, fld, agr_name, split_by_drug=True, verbose=verbose,
                                        use_unmatched=use_unmatched, filter_possible=True, possible=possibles)
    
    # now data is:
    # list for samples
    # for each sample: list of features for each drug
    ######

    # match: expect that all features in train set are in w2v model
    # may no longer be true if we filter w2v data and train data separately
    # feature may be acceptable for w2v but too rare in train
    train_vectors, train_names, unmatched_train = preproc(model, drugs, train, unmatched_train)
    test_vectors, test_names, unmatched_test = preproc(model, drugs, test, unmatched_test)

    if verbose:
        print('train', len(train_vectors), len(train_vectors[0]))
        print('test', len(test_vectors), len(test_vectors[0]))
        print('y_train', y_train.shape)
        print('y_test', y_test.shape)
        
    # for each drug train_vectors are a list of tensors
    train_vectors = [[torch.Tensor(x) for x in v] for v in train_vectors]
    y_train = [torch.Tensor(x) for x in y_train]
    y_train = torch.stack(y_train).long()

    test_vectors = [[torch.Tensor(x) for x in v] for v in test_vectors]
    y_test = [torch.Tensor(x) for x in y_test]
    y_test = torch.stack(y_test).long()
    if verbose:
        print('train', y_train.shape)
        print('test', y_test.shape)
    if return_names:
        if use_unmatched:
            return train_names, train_vectors, y_train, test_names, test_vectors, y_test, unmatched_train, unmatched_test
        else:
            return train_names, train_vectors, y_train, test_names, test_vectors, y_test
        #return train_samples, train_names, train_vectors, y_train, test_samples, test_names, test_vectors, y_test
        #return train_samples, train_vectors, y_train, test_samples, test_vectors, y_test
    if use_unmatched:
        return train_vectors, y_train, test_vectors, y_test, unmatched_train, unmatched_test
    else:
        return train_vectors, y_train, test_vectors, y_test


# updated for multi
def train_function(model, loss_fn, optimizer, X_train, y_train, X_test, y_test, unmatched_train, unmatched_test,  m_name, st=1, epochs=10, scheduler=None, save=0, f=lambda x:0, coef=1):
    model.train()
    sz = len(model.drugs)
    n = int(sz * (sz - 1) / 2)
    loss_test = np.zeros((sz, epochs))
    loss_train = np.zeros((sz, epochs))
    pen_train = np.zeros((n, epochs))
    pen_test = np.zeros((n, epochs))
    f1_test = np.zeros((sz, epochs))
    f1_train = np.zeros((sz, epochs))
    acc_test = np.zeros((sz, epochs))
    acc_train = np.zeros((sz, epochs))
    auc_test = np.zeros((sz, epochs))
    auc_train = np.zeros((sz, epochs))
    recall_test = np.zeros((sz, epochs))
    recall_train = np.zeros((sz, epochs))
    sm = nn.Softmax(dim=-1)
    if st != 1:
        print('NB: learning rate updated every', st, 'epochs')
    #print([[len(v) for v in vec] for vec in X_train])
    with tqdm(range(epochs)) as tbar:
        for ep in tbar:
            model.train()
            for i in range(len(X_train)):
                #print(i, [len(v) for v in X_train[i]])
                #print(i, [len(v) for v in X_train[i]])
                #if i in unmatched_train:
                #    print('Unmatched:', unmatched_train[i])
                #if i > 5:
                #  break
                # if model has attention, it returnes y_pred, attn_weights
                # attn is list of tensors (one tensor per drug)
                y_pred, attn = model(X_train[i])
                #loss, l_a = model.get_loss(y_pred.unsqueeze(0), y_train[i].unsqueeze(0), attn.unsqueeze(0), loss_fn, f=f, coef=coef)
                loss, l_a, pen = loss_fn(y_pred.unsqueeze(0), y_train[i].unsqueeze(0), [attn], {0: unmatched_train.get(i, dict())})
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
            if scheduler != None:
                if not ep % st:
                    scheduler.step()

            model.eval()
            with torch.no_grad():
                # metrics for test set
                res = [0] * len(X_test)
                attn = [0] * len(X_test)
                for i in range(len(X_test)):
                    res[i], attn[i] = model(X_test[i])
                    #res[i] = res[i].view(1, -1)
                res = torch.Tensor(np.array(res))
                #print(res.shape)
                for i in range(len(model.drugs)):
                    f1_test[i, ep] = f1_score(y_test[:, i], res[:, i, :].argmax(dim=-1))   
                    #f1_score(y_test[:, i], res[:, i].argmax(dim=-1))
                    acc_test[i, ep] = sum(res[:, i, :].argmax(dim=-1) == y_test[:, i]).item() / res.shape[0] 
                    #sum(res.argmax(dim=-1) == y_test[:, 0]).item() / res.shape[0]
                    auc_test[i, ep] = auc_score(y_test[:, i], sm(res[:, i, :])[:, -1])   
                    #auc(y_test[:, 0], sm(res)[:, 1])
                    recall_test[i, ep] = recall_score(y_test[:, i], res[:, i, :].argmax(dim=-1))  
                    #recall_score(y_test, res.argmax(axis=1))
                #loss, l_a = model.get_loss(res, y_test, attn, loss_fn, f=f, coef=coef)
                loss, l_a, pen = loss_fn(res, y_test, attn, unmatched_test)
                loss_test[:, ep] = l_a
                pen_test[:, ep] = pen
                # metrics for train set
                res = [0] * len(X_train)
                attn = [0] * len(X_train)
                for i in range(len(X_train)):
                    res[i], attn[i] = model(X_train[i])
                    #res[i] = res[i].view(1, -1)
                res = torch.Tensor(np.array(res))
                for i in range(len(model.drugs)):
                    f1_train[i, ep] = f1_score(y_train[:, i], res[:, i, :].argmax(dim=-1))   
                    #f1_score(y_test[:, i], res[:, i].argmax(dim=-1))
                    acc_train[i, ep] = sum(res[:, i, :].argmax(dim=-1) == y_train[:, i]).item() / res.shape[0]  
                    #sum(res.argmax(dim=-1) == y_test[:, 0]).item() / res.shape[0]
                    auc_train[i, ep] = auc_score(y_train[:, i], sm(res[:, i, :])[:, -1])   
                    #auc(y_test[:, 0], sm(res)[:, 1])
                    recall_train[i, ep] = recall_score(y_train[:, i], res[:, i, :].argmax(dim=-1))  
                    #recall_score(y_test, res.argmax(axis=1))
                #loss, l_a = model.get_loss(res, y_train, attn, loss_fn, f=f, coef=coef)
                loss, l_a, pen = loss_fn(res, y_train, attn, unmatched_train)
                loss_train[:, ep] = l_a
                pen_train[:, ep] = pen
            if save:
                if not ep % save and ep > 0:
                    torch.save(model.state_dict(), f'lstm_models/{"_".join([x[:3] for x in model.drugs])}_{model.fld}_training_{m_name}_ep{ep}.pt')
            s = 'loss train: '
            for i in range(len(model.drugs)):
                s += f'{loss_train[i, ep]:.4}|'
            s = s[:-1]
            s += ' loss test: '
            for i in range(len(model.drugs)):
                s += f'{loss_test[i, ep]:.4}|'
            s = s[:-1]
            s += f' penalty train: '
            for i in range(n):
                s += f'{pen_train[i, ep]:.4}|'
            s = s[:-1]
            s += f' penalty test:'
            for i in range(n):
                s += f'{pen_test[i, ep]:.4}|'
            s = s[:-1]
            s += ' f1 train: '
            for i in range(len(model.drugs)):
                s += f'{f1_train[i, ep]:.2}|'
            s = s[:-1]
            s += ' f1 test: '
            for i in range(len(model.drugs)):
                s += f'{f1_test[i, ep]:.2}|'
            s = s[:-1]
            s += ' recall train: '
            for i in range(len(model.drugs)):
                s += f'{recall_train[i, ep]:.2}|'
            s = s[:-1]
            s += ' recall test: '
            for i in range(len(model.drugs)):
                s += f'{recall_test[i, ep]:.2}|'
            s = s[:-1]
            tbar.set_postfix_str(s)
    metrics = dict()
    metrics['f1_test'] = f1_test
    metrics['f1_train'] = f1_train
    metrics['loss_train'] = loss_train
    metrics['loss_test'] = loss_test
    metrics['acc_test'] = acc_test
    metrics['acc_train'] = acc_train
    metrics['auc_test'] = auc_test
    metrics['auc_train'] = auc_train
    metrics['recall_test'] = recall_test
    metrics['recall_train'] = recall_train
    metrics['penalty_test'] = pen_test
    metrics['penalty_train'] = pen_train
    return metrics


# updated for multi
def plot_metrics(metrics, drugs, w2v_len, mdl, outfile):
    n = metrics[list(metrics.keys())[0]].shape[1]
    x = np.linspace(1, n, n)
    fig, ax = plt.subplots(1, 5, figsize=(15, 3))
    #ax[0].plot(x, metrics['loss'])
    #ax[0].set_title('Loss')
    mt = ['loss',  'acc', 'recall', 'f1', 'auc']
    for i in range(len(mt)):
        for j in range(len(drugs)):
            # legend only for first feature, which is loss
            if not i:
                ax[i].plot(x, metrics[mt[i] + '_train'][j], label=drugs[j]+' train')
                ax[i].plot(x, metrics[mt[i] + '_test'][j], label=drugs[j]+' test')
                #if not j:
                #    ax[i].plot(x, metrics['penalty_train'], label='train penalty')
                #    ax[i].plot(x, metrics['penalty_test'], label='test penalty')
            else:
                ax[i].plot(x, metrics[mt[i] + '_train'][j])
                ax[i].plot(x, metrics[mt[i] + '_test'][j])
        ax[i].set_title(mt[i])
    fig.legend(bbox_to_anchor=(0.5, -0.15 * len(drugs)), loc='lower center', ncols=len(drugs))
    fig.suptitle(f'{", ".join(drugs)}, w2v {w2v_len}, {mdl}')
    plt.tight_layout()
    plt.savefig(outfile, dpi=250, bbox_inches='tight')
    plt.close()
    # plot penalty
    n = len(metrics['penalty_test'][0])
    x = np.linspace(1, n, n)
    sz = len(metrics['penalty_test'])
    fig, ax = plt.subplots(1, sz, figsize=(sz*4, 4))
    drug_ind = -1
    for d2 in range(len(drugs)):
        for d1 in range(d2):
            drug_ind += 1
            if not drug_ind:
                ax[drug_ind].plot(x, metrics['penalty_train'][drug_ind], label='Penalty on train set')
                ax[drug_ind].plot(x, metrics['penalty_test'][drug_ind], label='Penalty on test set')
            else:
                ax[drug_ind].plot(x, metrics['penalty_train'][drug_ind])
                ax[drug_ind].plot(x, metrics['penalty_test'][drug_ind])
    
            ax[drug_ind].set_title(drugs[d1] + ' and ' + drugs[d2])
    fig.legend(bbox_to_anchor=(0.5, 0), loc='upper center', ncols=2)
    fig.suptitle(f'{", ".join(drugs)}, w2v {w2v_len}, {mdl} penalty')
    plt.tight_layout()
    outfile2 = outfile.split('.')[0] + '_penalty.' + outfile.split('.')[1]
    plt.savefig(outfile2, dpi=250, bbox_inches='tight')
