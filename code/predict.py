from functions import load_model, predict_and_save, load_all_data_from_index
import pandas as pd

# saves predictions to file

drugs = sorted(['Ethambutol', 'Isoniazid', 'Pyrazinamide', 'Rifampicin'])

out_file = 'predictions.txt'

# for loading data from index
feature_lists = 'data/'

w2v_name = 'model'

cur_name = ''

hidden_dim = 16
bidir = True
positional = True
n = 1e8

print(drugs)
print('w2v:', w2v_name)

if positional:
    print('Positional encoding; n =', n)

print('File', out_file)

data_vectors, y_data, samples = load_all_data_from_index(drugs, w2v_name, feature_lists, cur_name, positional=positional, n=n)

print('Loaded data')

print(len(samples), 'samples', len(data_vectors), 'vectors')

cols = ['EMB_pred', 'EMB_prob_pos', 'INH_pred', 'INH_prob_pos', 'PYR_pred', 'PYR_prob_pos', 'RIF_pred', 'RIF_prob_pos']

multi_name = 'model'

print()

print('Model:', multi_name, end=' ')


model = load_model(drugs, hidden_dim, multi_name, bidir, multi=True)
model.eval()

res = predict_and_save(data_vectors, model)

data = pd.DataFrame(res, index=samples, columns=cols)

print('Done')


for cl in data.columns:
    if cl.find('pred') != -1:
        data[cl] = data[cl].astype(int)

data.to_csv(out_file)
