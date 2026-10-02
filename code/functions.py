import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import time
from gensim.models import Word2Vec
from tqdm import tqdm
from w2v_lstm_functions import make_feature_list, sort_key
from w2v_lstm_functions import preproc as preproc_single
from sklearn.metrics import roc_auc_score as auc_score
# roc_auc_score(y_true, y_pred), y_pred is probability of the greater class
from sklearn.metrics import f1_score, recall_score
import os
from scipy.sparse import csc_matrix
from scipy.spatial.distance import pdist, squareform

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from sklearn.metrics import jaccard_score

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
            msk = weights[i] > thr
            msk = msk.squeeze(-1)
            # only keep features that pass significance threshold
            weights[i] = weights[i][msk]
            #
            out[i] = out[i][msk]
            #weights[i] = weights[i].masked_fill(weights[i] < thr, 0)
        x = [0] * len(self.drugs)
        for i in range(len(self.drugs)):
          w = out[i] * weights[i]
          #print(w.shape)
          w = w.sum(dim=0)
          x[i] = self.linear[i](w)
        return torch.stack(x), weights


def load_model(drugs, hidden_dim, model_name, bidir, path='', skip_ind=[], use_attn=True, use_linear=True, empty=False, zero=False, mul=1):
    """
    Load saved model.

    multi: bool, default:True; True: Load multi-drug model, False: load model for a single drug

    skip_ind: list of indeces; for these drugs model is not loaded. Only used if multi=True.

    ep: list, epoch numbers for each model. If multi - list length of 1.

    use_attn: bool; if True, load attention weights
    use_linear: bool; only used if use_attn=True

    empty: create model w/o loading weights
    """
    if empty:
        model = lstm_attn_multi(drugs, 1, 100, hidden_dim, m_name=model_name, num_layers=1, 
                            bidirectional=bidir, zero=zero, mul=mul)
        return model
    
    model = lstm_attn_multi(drugs, 1, 100, hidden_dim, m_name=model_name, num_layers=1, 
                        bidirectional=bidir, zero=zero, mul=mul)
    
    model.load_state_dict(torch.load(f'data/lstm_{model_name}.pt'))
    # if we don't need pre-trained attention & linear layers
    if not use_attn or not use_linear:
        m = lstm_attn_multi(drugs, 1, 100, hidden_dim, m_name=model_name, num_layers=1, bidirectional=bidir, zero=zero, mul=mul)
        if not use_attn:
            for i in range(len(drugs)):
                model.attn[i].state_dict()['attn.weight'][:] = m.attn[i].state_dict()['attn.weight'][:]
        if not use_linear:
            for i in range(len(drugs)):
                model.linear[i].state_dict()['weight'][:] = m.linear[i].state_dict()['weight'][:]
                model.linear[i].state_dict()['bias'][:] = m.linear[i].state_dict()['bias'][:]
        # if not pre-trainde weights for some drugs
        if len(skip_ind):
            m = lstm_attn_multi(drugs, 1, 100, hidden_dim, m_name=model_name, num_layers=1, bidirectional=bidir, zero=zero, mul=mul)
            for ind in skip_ind:
                print('Skipping pre-trained model for', drugs[ind])
                for w in ['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0']:
                    model.lstm[ind].state_dict()[w][:] = m.lstm[ind].state_dict()[w][:]
                if bidir:
                    for w in ['weight_ih_l0_reverse', 'weight_hh_l0_reverse', 'bias_ih_l0_reverse', 'bias_hh_l0_reverse']:
                        model.lstm[ind].state_dict()[w][:] = m.lstm[ind].state_dict()[w][:]
                model.attn[ind].state_dict()['attn.weight'][:] = m.attn[ind].state_dict()['attn.weight'][:]
                model.linear[ind].state_dict()['weight'][:] = m.linear[ind].state_dict()['weight'][:]
                model.linear[ind].state_dict()['bias'][:] = m.linear[ind].state_dict()['bias'][:]

    model.eval()
    return model


def load_data_from_index(path, samples=[], name='', verbose=False, return_all=False, return_samples=False):
    """
    Construct feature lists from index files.
    Should be used for w2v training instead of prep().

    samples: list of IDs; if [] - loads data for all available samples
    return_all: bool, if True: returns list of all features in addition to lists of features for each sample.
    return_samples: bool, if True: returns list of all samples
    """
    # read possible features from file
    with open(os.path.join(path, f'all_features{name}.txt')) as f:
        all_features = np.array(f.readlines()[0].split())
    
    # read indeces from file
    with open(os.path.join(path, f'features{name}.txt')) as f:
        lines = [x.split() for x in f.readlines()]
    
    # first in line is sample ID
    if not len(samples):
        samples = [x[0] for x in lines]
    
    present_samples = [x[0] for x in lines]
    absent = [x for x in samples if x not in present_samples]
    if len(absent):
        print('WARNING: no data for', len(absent), 'samples:', absent)
    
    # only keep samples that have features in them
    samples = [x for x in samples if x in present_samples]

    if verbose:
        print('Samples:', len(samples))
        print('Features:', len(all_features))

    #skip = ['rplC_PF00297.28_Eth_Iso_Pyr_Rif']
    features = [0] * len(samples)
    # turn feature_lists of indeces back to list of strings
    for i in range(len(lines)):
        # check if we need this sample
        if lines[i][0] in samples:
            # index of sample in i-th line
            smpl = samples.index(lines[i][0])
            # indeces of features for that sample
            index = [int(x) for x in lines[i][1:]]
            # list of features for this sample
            features[smpl] = all_features[index].tolist()
            if verbose:
                print(samples[smpl], len(features[smpl]), 'features')
            if not len(features[smpl]):
                print('WARNING:', len(features[smpl]), 'features for', samples[smpl])
    if verbose:
        print('\nExample:', samples[0], len(features[0]), features[0][:10])
    if return_all:
        if return_samples:
            return samples, features, all_features
        else:
            return features, all_features
    if return_samples:
        return samples, features
    return features


def unmatch_agr(features, drugs, use_unmatched=False, verbose=False, verbose_debug=False):
    """
    Split agr features by drug, build unmatched

    paramteters:
    features: list of lists of string, given features for each sample
    drugs: list of strings, drugs
    """    
    
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
                # keep all regular features, keep domain features if they are for the right drug
                res[i][d] = [x for x in features[i] if x.find('_PF') == -1 or x.find('_PF') != -1 and x.find(drugs[d][:3]) != -1]
                # list of domain features for this drug
                fts[d] = [x for x in res[i][d] if x.find('_PF') != -1]
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
                    if verbose_debug:
                        print(drugs[d2], drugs[d], sorted(list(s)))
                    # unmatched features for both drugs
                    unm1 = [res[i][d2].index(x) for x in fts[d2] if x[:x.find('.', x.find('PF'))] not in s]
                    unm2 = [res[i][d].index(x) for x in fts[d] if x[:x.find('.', x.find('PF'))] not in s]
                    # if either drug has unmatched features, save them to dict
                    if unm1 or unm2:
                        unmatched[i][(d2, d)] = [unm1, unm2]
                        # check that after matching the lengths are the same
                        if len(res[i][d2]) - len(unmatched[i][(d2, d)][0]) != len(res[i][d]) - len(unmatched[i][(d2,d)][1]):
                            print('ERROR: lengths do not match after unmatching:', drugs[d2], drugs[d], i)
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
                # keep all regular features, keep domain features if they are for the right drug
                res[i][d] = [x for x in features[i] if x.find('_PF') == -1 or x.find('_PF') != -1 and x.find(drugs[d][:3]) != -1]
                # sort features
                res[i][d].sort(key=sort_key)
        return res, {}
    

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


def preprocess_data(data, model, drugs, set_name, positional=False, n=1e5, verbose=False, use_unmatched=False, return_names=False):
    """
    Turn list of features into vectors.
    data: list of features for each sample
    """
    if verbose:
        print('Unmatching domain features...')
    # unmatch_agr(features, drugs, use_unmatched=False, verbose=False, verbose_debug=False)
    data, unmatched_data = unmatch_agr(data, drugs, use_unmatched=use_unmatched, verbose=verbose)
    
    # now data is:
    # list for samples
    # for each sample: list of features for each drug (sorted along the genome)
    ######

    # match: expect that all features in train set are in w2v model
    # may no longer be true if we filter w2v data and train data separately
    # feature may be acceptable for w2v but too rare in train
    data_vectors, data_names, unmatched_data = preproc(model, drugs, data, unmatched_data)

    if verbose:
        print(set_name, len(data_vectors), len(data_vectors[0]))

    if positional:
        # tr[i][j] (i-sample, j - drug)
        # w2v dimension
        dim = data_vectors[0][0].shape[1]
        # names[i][d] = [x for x in data[i][d] if x in possibles]
        pos_enc = np.zeros(dim)
        # vectors and feature_names
        for s in range(len(data_vectors)):
            for d in range(len(data_vectors[s])):
                # list of features for this samples and drug
                fts = data_names[s][d]
                for f in range(len(fts)):
                    ft = fts[f]
                    coord = sort_key(ft)
                    for i in range(dim // 2):
                        pos_enc[2*i] = np.sin(coord / (n ** (2*i / dim)))
                        pos_enc[2*i+1] = np.cos(coord / (n ** (2*i / dim)))
                    # add positional encoding to feature
                    data_vectors[s][d][f] += pos_enc
    
    # for each drug train_vectors are a list of tensors
    data_vectors = [[torch.from_numpy(x.copy()) for x in v] for v in data_vectors]

    if return_names:
        if use_unmatched:
            return data_names, data_vectors, unmatched_data
        else:
            return data_names, data_vectors
        #return train_samples, train_names, train_vectors, y_train, test_samples, test_names, test_vectors, y_test
        #return train_samples, train_vectors, y_train, test_samples, test_vectors, y_test
    if use_unmatched:
        return data_vectors, unmatched_data
    else:
        return data_vectors


def construct_data_from_index(model, path, name, drugs, set_name, samples=[], positional=False, renew_samples=False, n=1e5, verbose=False, use_unmatched=False, return_names=False):
    """
    Turn list of samples and vector of y-values into data ready to go into model training.

    renew_samples: bool, if True: skip samples that do not have data for them and update list of samples.
    """
    # load_data_from_index(samples, path, name='', verbose=False)
    data = load_data_from_index(path, samples=samples, name=name, verbose=verbose, return_samples=renew_samples)

    #
    if renew_samples:
        samples_new, data = data
        if len(samples) == len(samples_new):
            samples = samples_new
        # if some samples were dropped
        else:
            samples = samples_new
    
    if verbose:
        print('Loaded data')
    
    data = preprocess_data(data, model, drugs, set_name, positional=positional, n=n, verbose=verbose, use_unmatched=use_unmatched, return_names=return_names)
    if renew_samples:
        return data, samples
    return data



def load_all_data_from_index(drugs, w2v_name, path, name, no_y=True, res_path='../data/all_resistance.csv', positional=False, n=1e5, verbose=False, use_unmatched=False, return_names=False):
    """
    Loads all samples from index, constructs data vectors ready to go into ml model.
    """

    model = Word2Vec.load(f"data/word2vec_{w2v_name}.model")

    print('Loading data from index')

    #data_names, data_vectors, y_data

    constr, samples = construct_data_from_index(model, path, name, drugs, 'all_samples', positional=positional, 
                                renew_samples=True, n=n, verbose=verbose, use_unmatched=use_unmatched, return_names=return_names)
    
    print(len(constr[-1]), 'vectors')

    if return_names:
        data_names, data_vectors = constr
        return data_names, data_vectors, samples
    else:
        data_vectors = constr
        return data_vectors, samples


def predict_and_save(data, model):
    model.eval()
    sm = nn.Softmax(dim=-1)

    with torch.no_grad():
        # predict
        res = [0] * len(data)
        for i in range(len(data)):
            res[i], attn = model(data[i])
            #res[i] = res[i].view(1, -1)
        res = torch.Tensor(np.array(res))
    # binary prediction
    pred = np.array([res[:, d, :].argmax(dim=-1) for d in range(len(res[0]))]).T
    prob = np.array([sm(res[:, d, :]) for d in range(len(res[0]))])
    prob = np.array([sm(res[:, d, :])[:, -1] for d in range(len(res[0]))]).T
    res = np.array([list(zip(pred[i], prob[i])) for i in range(len(pred))])
    res = res.reshape(res.shape[0], -1)
    return res
