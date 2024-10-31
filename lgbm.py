import pandas as pd
import numpy as np
import sys
from catboost import CatBoostRegressor
import lightgbm as lgb
from lightgbm import Dataset
sys.path.append('../..')
#from collect_data.preprocessing.preprocess_refactor import Path
from data.jalan.preprocessing import Path
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import sklearn.preprocessing as sp
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

if __name__=='__main__':
    path = Path()

    mask = np.load(path.valid_idx_path)
    train_mask = mask[mask>=2]
    val_mask = mask[mask==0]
    test_mask = mask[mask==1]
    
    df = pd.read_csv(path.df_experience_light_path)
    X = np.load(path.spot_img_emb_path)
    X = np.load('./data/jalan/graph/spot_img_emb_multi_ResNet.npy')
    X = np.mean(X.reshape(-1, 5, 512), axis=1)
    print(X.shape)
    le = LabelEncoder()
    df['city_label'] = le.fit_transform(df['city'])
    le = LabelEncoder()
    df['pref_label'] = le.fit_transform(df['prefecture'])
    oh = OneHotEncoder(sparse=False)
    place_labels = oh.fit_transform(df[['city_label','pref_label']])
    word_vec = get_wordvec()
    print(word_vec.shape)
    word_vec = np.mean(word_vec.reshape(42852, 15, 300), axis=1)
    X = np.concatenate([X,word_vec], 1)
    print(X.shape)
    y = np.log10(df['review_count']).values

    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]

    cat_features = [i for i in range(512, X_train.shape[1])]
    #model= CatBoostRegressor(iterations=1000,learning_rate=1, depth=5, loss_function='RMSE')
    model = lgb.LGBMRegressor(
            random_state = 71,
            n_estimators=1000
        )
    model.fit(X_train, y_train, 
              eval_set=(X_val, y_val),
              eval_metric='mse',
              early_stopping_rounds=50,
              verbose=True)
    pred = model.predict(X_test)
    print(pred.shape, print(y_test.shape))
    print(y, pred)
    print(np.corrcoef(pred, y_test))

