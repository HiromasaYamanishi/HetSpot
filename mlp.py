import pandas as pd
import numpy as np
import sys
from catboost import CatBoostRegressor
import lightgbm as lgb
import sklearn.preprocessing as sp
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
sys.path.append('../..')
#from collect_data.preprocessing.preprocess_refactor import Path
from data.jalan.preprocessing import Path

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', 2048)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

class EarlyStopping:
    """earlystoppingクラス"""

    def __init__(self, patience=5, verbose=False, path='checkpoint/checkpoint_model.pth'):
        """引数：最小値の非更新数カウンタ、表示設定、モデル格納path"""

        self.patience = patience    #設定ストップカウンタ
        self.verbose = verbose      #表示の有無
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        self.val_loss_min = np.Inf   #前回のベストスコア記憶用
        self.path = path             #ベストモデル格納path

    def __call__(self, val_loss, model):
        """
        特殊(call)メソッド
        実際に学習ループ内で最小lossを更新したか否かを計算させる部分
        """
        score = -val_loss

        if self.best_score is None:  #1Epoch目の処理
            self.best_score = score   #1Epoch目はそのままベストスコアとして記録する
            self.checkpoint(val_loss, model)  #記録後にモデルを保存してスコア表示する
        elif score < self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            if self.verbose:  #表示を有効にした場合は経過を表示
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する 
            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
                #torch.save(self.best_model, self.path)
                self.early_stop = True
        else:  #ベストスコアを更新した場合
            self.best_score = score  #ベストスコアを上書き
            self.best_model = model.state_dict()
            self.checkpoint(val_loss, model)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット

    def checkpoint(self, val_loss, model):
        '''ベストスコア更新時に実行されるチェックポイント関数'''
        if self.verbose:  #表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #torch.save(model.state_dict(), self.path)  #ベストモデルを指定したpathに保存
        self.val_loss_min = val_loss  #その時のlossを記録する
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

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=200):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
                    nn.Linear(input_size*1, self.hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        y = self.net(x)
        return y

if __name__=='__main__':
    path = Path()
    df = pd.read_csv(path.df_experience_path)
    print(df.columns.values)
    y = np.log10(df['review_count']).values
    image_emb = np.load('./data/graph/spot_img_emb_multi_ResNet.npy')
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
    sc = np.load('./data/graph/spot_category.npy')
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
    X = np.concatenate([image_emb,word_vec], 1)
    y = np.log10(df['review_count']).values
    device = 'cuda'
    X = torch.tensor(X).to(device).float()
    y = torch.tensor(y).to(device).float()
    valid_id = np.load(path.valid_idx_path)
    train_mask,val_mask,test_mask = (valid_id>1), (valid_id==0), (valid_id==1)
    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]
    model = MLP(X_train.shape[1])
    model = model.to(device)
    earlystopping = EarlyStopping(patience=200, verbose=True)

    epoch = 1000
    optimizer = Adam(model.parameters(), lr=1e-4)

    for i in range(epoch):
        for phase in ['train','val']:
            total_loss = 0
            if phase=='train':
                for ind in minibatch(np.arange(X_train.shape[0])):
                    model.train()
                    out = model(X_train[ind])
                    loss = F.mse_loss(out, y_train[ind])
                    total_loss+=loss
                    loss.backward()
                    optimizer.step()
            else:
                model.eval()
                out = model(X_val)
                loss = F.mse_loss(out, y_val)
                earlystopping(loss, model) #callメソッド呼び出し
            
            print(f'epoch: {i} phase:{phase}, loss:{loss}')
        if earlystopping.early_stop: #ストップフラグがTrueの場合、breakでforループを抜ける
            print("Early Stopping!")
            
    #model.load_state_dict(torch.load('checkpoint_model.pth'))
    model.eval()
    pred = model(X_test)
    #print(model(X_val))
    print(pred.shape, y_test.shape)
    #print(y_test.flatten().shape)
    print(pred)
    print(np.corrcoef(y_test.cpu().detach().numpy(),pred.flatten().cpu().detach().numpy()))

