from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
from utils import save_cor
import sys
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from torchmetrics import MeanAbsolutePercentageError
import torch
import math
sys.path.append('../..')
#from collect_data.preprocessing.preprocess_refactor import Path
from data.jalan.preprocessing import Path

from sklearn.neural_network import MLPRegressor 
def get_wordvec():
    path = Path()
    word_embs=np.load(path.word_embs_finetune_path)
    
    word_indexs = np.load(path.tfidf_topk_index_path)
    top_words = np.load(path.tfidf_top_words_path)
    tfidf_words = np.load(path.tfidf_word_path)
    word_vec_all= []
    
    for ind in word_indexs:
        word_vec_all.append(np.concatenate(word_embs[ind]))
    word_vec_all = np.array(word_vec_all)
    return word_vec_all

def get_coef(X, model):
    valid_id = np.load(path.valid_idx_path)
    train_mask = (valid_id>1)
    val_mask = (valid_id==0)
    test_mask = (valid_id==1)
    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]


    model.fit(X_train, y_train)
    pred = clf.predict(X_test)
    mse = torch.nn.MSELoss()
    mape = MeanAbsolutePercentageError().to('cpu')
    mae = torch.nn.L1Loss()
    mse = mse(torch.from_numpy(pred), torch.from_numpy(y_test)).item()
    rmse = math.sqrt(mse)
    mape = mape(torch.from_numpy(pred), torch.from_numpy(y_test))
    mae = mae(torch.from_numpy(pred), torch.from_numpy(y_test))
    print('corr', np.corrcoef(y_test, pred)[0][1], 'mse', mse, 'rmse', rmse, 'mape', mape.item(), 'mae', mae.item())
    return np.corrcoef(y_test, pred)[0][1]
    #print(np.corrcoef(y_test, pred)[0][1])

if __name__ == '__main__':
    path = Path()
    df = pd.read_csv(path.df_experience_path)
    print(df.columns.values)
    y = np.log10(df['review_count']).values
    image_emb = np.load(path.spot_img_emb_path)
    image_emb = np.load('/home/yamanishi/project/trip_recommend/data/jalan/graph/spot_img_emb_multi_ResNet.npy')
    image_emb = np.mean(image_emb.reshape(-1, 5, 512), axis=1)
    print(image_emb.shape)
    le = LabelEncoder()
    #df['category_label'] = le.fit_transform(df['category'])
    le = LabelEncoder()
    #df['city_label'] = le.fit_transform(df['city'])
    le = LabelEncoder()
    #df['pref_label'] = le.fit_transform(df['都道府県'])
    #oh = OneHotEncoder(sparse=False)
    #place_labels = oh.fit_transform(df[['category_label','city_label','pref_label']])
    oh = OneHotEncoder(sparse=False)
    sc = np.load('./data/jalan/graph/spot_category.npy')
    category_label = np.zeros((len(df), sc[1].max()+1))
    category_label[sc[0], sc[1]] = 1
    print(category_label.shape)
    city_label = oh.fit_transform(df[['city']])
    print(city_label.shape)
    oh = OneHotEncoder(sparse=False)
    pref_label = oh.fit_transform(df[['prefecture']])
    print(pref_label.shape)
    word_vec = get_wordvec()
    word_vec = np.mean(word_vec.reshape(42852, 15, 300), axis=1)
    print(word_vec.shape)
    print(image_emb.shape, category_label.shape, city_label.shape, pref_label.shape, word_vec.shape)
    hidden_layer_sizes_all = [(100, 100), (100, 100, 100), (100, 100, 100, 100)]
    
    #for hidden_layer_sizes in hidden_layer_sizes_all:
    hidden_layer_sizes = (100, 100, 100)
    for i in range(3):
        X = np.concatenate([image_emb,word_vec], 1)
        clf = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes)
        coef = get_coef(X, clf)
        #print(coef)
    for i in range(3):
        X = np.concatenate([image_emb,word_vec, category_label], 1)
        clf = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes)
        coef = get_coef(X, clf)
        #print(coef)
    for i in range(3):
        X = np.concatenate([image_emb,word_vec, category_label, city_label], 1)
        clf = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes)
        coef = get_coef(X, clf)
        #print(coef)
        # X = np.concatenate([image_emb,word_vec, category_label, city_label, pref_label], 1)
        # clf = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes)
        # coef = get_coef(X, clf)
        #print(coef)
        print('X shape', X.shape)

    