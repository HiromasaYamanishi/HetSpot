from this import d
from typing import Union
import os
import time
import sys
from PIL import Image
from requests_html import HTMLSession
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
import gensim
import json
import pickle
import re
from janome.analyzer import Analyzer
from janome.tokenfilter import POSKeepFilter,TokenCountFilter, CompoundNounFilter
from janome.charfilter import RegexReplaceCharFilter, UnicodeNormalizeCharFilter
import torch
import torchvision
from torchvision import models
from torchvision import transforms as transforms
from torch import nn
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from collections import defaultdict
tqdm.pandas()
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"
def get_key_from_value(d, val):
    return [k for k,v in d.items() if v==val][0]

class Path:
    def __init__(self):
        self.df_experience_path = './data/spot/experience.csv'
        self.df_experience_light_path = './data/spot/experience_light.csv'
        self.df_review_path = './data/review/review_all.csv'
        #self.df_review = pd.read_csv(self.df_review_path)

        self.data_graph_dir = '/data/graph'
        self.data_dir = './data'
        self.flickr_image_dir = './data/flickr_image'
        self.jalan_image_dir = './data/jalan_image'
        self.category_image_dir = './data/category_image'

        self.valid_idx_path = os.path.join(self.data_graph_dir, 'valid_idx.npy')
        self.spot_index_path = os.path.join(self.data_graph_dir,'spot_index.pkl')
        self.index_spot_path = os.path.join(self.data_graph_dir,'index_spot.pkl')
        self.index_word_path = os.path.join(self.data_graph_dir,'index_word.pkl')
        self.word_index_path = os.path.join(self.data_graph_dir,'word_index.pkl')
        self.city_index_path = os.path.join(self.data_graph_dir,'city_index.pkl')
        self.index_city_path = os.path.join(self.data_graph_dir,'index_city.pkl')
        self.pref_index_path = os.path.join(self.data_graph_dir,'pref_index.pkl')
        self.index_pref_path = os.path.join(self.data_graph_dir,'index_pref.pkl')
        self.category_index_path = os.path.join(self.data_graph_dir, 'category_index.pkl')
        self.index_category_path = os.path.join(self.data_graph_dir, 'index_category.pkl')
        self.tfidf_topk_index_path = os.path.join(self.data_graph_dir, 'tfidf_topk_index.npy')
        self.tfidf_top_words_path = os.path.join(self.data_graph_dir, 'tfidf_top_words.npy')
        self.tfidf_word_path = os.path.join(self.data_graph_dir,'tfidf_words.npy')
        self.tfidf_topk_index_spare_path = os.path.join(self.data_graph_dir, 'tfidf_topk_index_spare.npy')
        self.tfidf_top_words_spare_path = os.path.join(self.data_graph_dir, 'tfidf_top_words_spare.npy')
        self.tfidf_word_spare_path = os.path.join(self.data_graph_dir,'tfidf_words_spare.npy')
        self.tfidf_word_th_path = os.path.join(self.data_graph_dir,'tfidf_words_th.npy')
        self.word_popularity_path = os.path.join(self.data_graph_dir, 'word_popularity.npy')
        self.word_embs_path = os.path.join(self.data_graph_dir,'word_embs.npy')
        self.word_embs_th_path = os.path.join(self.data_graph_dir, 'word_embs_th.npy')
        self.word_embs_wiki_path = os.path.join(self.data_graph_dir,'word_embs_wiki.npy')
        self.word_embs_finetune_path = os.path.join(self.data_graph_dir,'word_embs_finetune.npy')
        self.word_embs_ensemble_path = os.path.join(self.data_graph_dir,'word_embs_ensemble.npy')
        self.word_emb_clip_path = os.path.join(self.data_graph_dir,'word_emb_clip.npy')
        self.spot_word_path = os.path.join(self.data_graph_dir,'spot_word.npy')
        self.spot_word_th_path = os.path.join(self.data_graph_dir,'spot_word_th.npy')
        self.spot_category_path = os.path.join(self.data_graph_dir, 'spot_category.npy')
        self.spot_popularity_path = os.path.join(self.data_graph_dir, 'spot_popularity.npy')
        self.spot_city_path = os.path.join(self.data_graph_dir, 'spot_city.npy')
        self.city_pref_path = os.path.join(self.data_graph_dir, 'city_pref.npy')
        self.city_adj_path = os.path.join(self.data_graph_dir, 'city_adj.pkl')
        self.city_city_path = os.path.join(self.data_graph_dir, 'city_city.npy')
        self.city_popularity_path = os.path.join(self.data_graph_dir, 'city_popularity.npy')
        self.pref_popularity_path = os.path.join(self.data_graph_dir, 'pref_popularity.npy')
        self.pref_pref_path = os.path.join(self.data_graph_dir, 'pref_pref.npy')
        self.pref_attr_cat_path = os.path.join(self.data_graph_dir, 'pref_attr_cat.npy')
        self.spot_pref_path = os.path.join(self.data_graph_dir, 'spot_pref.npy')
        self.spot_spot_path = os.path.join(self.data_graph_dir, 'spot_spot.npy')
        self.pref_attr_path = os.path.join(self.data_graph_dir, 'pref_attr.npy')
        self.city_attr_path = os.path.join(self.data_graph_dir, 'city_attr.npy')
        self.city_attr_cat_path = os.path.join(self.data_graph_dir, 'city_attr_cat.npy')
        self.spot_img_emb_path = os.path.join(self.data_graph_dir, 'spot_img_emb_ResNet.npy')
        self.category_img_emb_path = os.path.join(self.data_graph_dir, 'category_img_emb.npy')
        self.category_emb_path = os.path.join(self.data_graph_dir, 'category_emb.npy')
        self.spot_img_emb_multi_path = os.path.join(self.data_graph_dir,'spot_img_emb_multi.npy')
        self.spot_img_emb_clip_path = os.path.join(self.data_graph_dir, 'spot_img_emb_clip.npy')

class Makedf(Path):
    def __init__(self):
        super().__init__()

    def join_review(self,df):
        #df = self.df_spot_wikipedia
        df_review = pd.read_csv(self.df_review_path)
        for i,spot in enumerate(tqdm(df['spot_name'])):
            df_review_tmp = df_review[df_review['spot'] == spot]
            review_all = ''
            for review in df_review_tmp['review']:
                review_all += review

            df.loc[i, 'jalan_review'] = review_all
        #df.to_csv(self.df_experience_path)
        return df
        
    def join_text(self, df):
        #df = self.df_experience
        df['text_joined']= df['description'] +' ' +df['jalan_review']
        #df.to_csv(self.df_experience_path)
        return df
    
    def drop_na(self, df):
        df = df.dropna()
        return df

    def make_experience_df(self, df):
        print('making experience df')

        df = self.df_spot_wikipedia
        df_text = pd.read_csv(self.df_spot_wikipedia_text_path)
        df[['wiki_text']]=df_text[['wiki_text']]
        df = self.join_review(df)
        print('join review done')

        df = self.join_text(df)
        print('join text done')

        for col in df.columns.values:
            if 'Unnamed' in col:
                df=df.drop(col, axis=1)
        if 'state' in df.columns.values:
            df = df.drop('state', axis=1)
        df = df.dropna()
        df = df.reset_index(drop=True)

        df = self.get_stratified_fold_split(df)
        print('stratify done')

        df.to_csv(self.df_experience_path)
        print('experience df done')
        return df

    def get_stratified_fold_split(self, df, n_folds=10, q=20):
        print('start stratified fold')
        #df = self.df_experience
        df['review_count_rank']=df['review_count'].rank(method='first')
        y = df['review_count_rank'].values
        X = df['spot_name'].values
        y_cat = pd.qcut(y, q=q).codes
        fold = StratifiedKFold(n_splits=n_folds, random_state=73, shuffle=True)
        array = np.array([0]*len(df))
        for i,(train_idx, valid_idx) in enumerate(fold.split(X, y_cat)):
            array[valid_idx] = i
        np.save(self.valid_idx_path, array)
        df['valid'] = pd.Series(array)
        #df.to_csv(self.df_experience_path)
        #self.df_experience = df
        print('end stratified fold')
        return df

    def parse_geocode_pickle(self):
        df = pd.read_csv(self.df_experience_path)
        for i in range(len(df)):
            spot_name = df.loc[i,'spot_name']
            print(spot_name)
            pickle_path = f'./data/gmap/{spot_name}.pkl'
            if os.path.exists(pickle_path):
                with open(f'./data/gmap/{spot_name}.pkl','rb') as f:
                    result = pickle.load(f)
                    #print(result)
                    if len(result)>0:
                        shi = [v for v in result[0]['address_components'] if v['types']==['locality', 'political']]
                        gun = [v for v in result[0]['address_components'] if v['types']==['administrative_area_level_2', 'political']]
                        print(spot_name, shi, gun)
                        if len(shi)==0:continue
                        city=shi[0]['long_name']
                        df.loc[i,'city']=city
        df.to_csv(self.df_experience_path)

class Tokenize(Path):
    def __init__(self):
        super().__init__()

    def tokenize_text(self, df,column='text_joined', hinshi=['名詞','形容詞','動詞']):
        '''
        columnのtextをトークナイズする
        '''
        print('start tokenize')
        #df = self.df_experience
        def join_nouns(text):
            if pd.isna(text):
                return None
            char_filters = [
                UnicodeNormalizeCharFilter(),
                RegexReplaceCharFilter('[#!:;<>{}・`.,()-=$/_\d\'"\[\]\|年月日~]+', ' '),
            ]
            token_filters = [
                POSKeepFilter(hinshi),
                #CompoundNounFilter()
                #TokenCountFilter(),
            ]
            analyzer = Analyzer(char_filters = char_filters, token_filters = token_filters)
            token_nouns = analyzer.analyze(text)
            token_nouns = [l.surface for l in token_nouns]
            joint_nouns = (' ').join(token_nouns)
            return joint_nouns
        df['tokenized_text']=df[column].progress_apply(join_nouns)
        #df.to_csv(self.df_experience_path)
        #self.df_experience = df
        print('end tokenize')
        return df

    def clean_text(self, df):
        def clean_text_(text):
            if isinstance(text, float):
                return None
            for word in [ '済み ', 'net ', 'コロナ ', '済み', 'net', 'コロナ']:
                text = text.replace(word, '')
            return text

        df['tokenized_text_clean'] = df['tokenized_text'].progress_apply(clean_text_)
        print('end cleaning text')
        return df


class TFIDF(Path):
    def __init__(self):
        super().__init__()

    def tfidf_threshold(self, df, column='tokenized_text', threshold=0.1, save= True, return_words=False):
        '''
        columnの単語列をtf-idfする
        '''
        print('start tfidf_threshold')
        #df = self.df_experience
        df=df.fillna({'tokenized_text':''})
        tfidf = TfidfVectorizer(min_df=3,max_df=0.5)
        tfidf_vec = tfidf.fit_transform(df[column]).toarray()
        feature_names = np.array(tfidf.get_feature_names())
        spot_names = df['spot_name'].values
        if save== True:
            np.save(self.tfidf_word_th_path, feature_names)
        edges = np.array(np.where(tfidf_vec>threshold))
        print(edges.shape)
        np.save(self.spot_word_th_path, edges)
        print('end tfidf threshold')
        return df

    def tfidf_topk(self, df, column='tokenized_text', k=15, save= True, return_words=False):
        '''
        columnの単語列をtf-idfする
        '''
        print('start tfidf')
        #df = self.df_experience
        df=df.fillna({column:''})
        if 'tfidf_topk_index' not in df.columns.values:
            df['tfidf_topk_index'] = None
        if 'tfidf_topk_word' not  in df.columns.values:
            df['tfidf_topk_word'] = None
        df['tfidf_topk_index'] = df['tfidf_topk_index'].astype(object)
        df['tfidf_topk_word'] = df['tfidf_topk_word'].astype(object)
        tfidf = TfidfVectorizer(min_df=3,max_df=0.5)
        tfidf_vec = tfidf.fit_transform(df[column]).toarray()
        print(tfidf_vec.shape)
        feature_names = np.array(tfidf.get_feature_names())
        if save== True:
            np.save(self.tfidf_word_path, feature_names)
        inds = []
        for i in range(len(df)):
            inds.append(np.argsort(-tfidf_vec[i])[:k])
        print(len(inds))
        if save==True:
            np.save(self.tfidf_topk_index_path, inds)
        tfidf_top_words = []
        for i,ind in enumerate(inds):
            if df.loc[i,column]!='':
                df.at[i, 'tfidf_topk_index'] = ind
                df.at[i, 'tfidf_topk_word'] = feature_names[np.array(ind)]
                tfidf_top_words.append(feature_names[np.array(ind)])
                #print(i,df.loc[i,'spot_name'],ind)
                #print(feature_names[np.array(ind)])
        if save==True:
            np.save(self.tfidf_top_words_path, tfidf_top_words)
        #df.to_csv(self.df_experience_path)
        #self.df_experience = df
        print('end tfidf')
        if return_words==True:
            return df, tfidf_top_words
        else:
            return df

    def tfidf_topk_spare(self, df, column='tokenized_text', k=15, save= True, return_words=False):
        '''
        columnの単語列をtf-idfする
        '''
        print('start tfidf')
        #df = self.df_experience
        df=df.fillna({column:''})
        if 'tfidf_topk_index' not in df.columns.values:
            df['tfidf_topk_index'] = None
        if 'tfidf_topk_word' not  in df.columns.values:
            df['tfidf_topk_word'] = None
        df['tfidf_topk_index'] = df['tfidf_topk_index'].astype(object)
        df['tfidf_topk_word'] = df['tfidf_topk_word'].astype(object)
        tfidf = TfidfVectorizer(min_df=3,max_df=0.5)
        tfidf_vec = tfidf.fit_transform(df[column]).toarray()
        with open('./data/jalan/graph/tfidf_vec.pkl', 'wb') as f:
            pickle.dump(tfidf, f)
        print(tfidf_vec.shape)
        feature_names = np.array(tfidf.get_feature_names())
        if save== True:
            np.save(self.tfidf_word_spare_path, feature_names)
        inds = []
        for i in range(len(df)):
            inds.append(np.argsort(-tfidf_vec[i])[:k])
        print(len(inds))
        if save==True:
            np.save(self.tfidf_topk_index_spare_path, inds)
        tfidf_top_words = []
        for i,ind in enumerate(inds):
            if df.loc[i,column]!='':
                df.at[i, 'tfidf_topk_index'] = ind
                df.at[i, 'tfidf_topk_word'] = feature_names[np.array(ind)]
                tfidf_top_words.append(feature_names[np.array(ind)])
        if save==True:
            np.save(self.tfidf_top_words_spare_path, tfidf_top_words)
        print('end tfidf')
        if return_words==True:
            return df, tfidf_top_words
        else:
            return df


class ConstructGraph(Path):
    def __init__(self):
        super().__init__()

    def spot_index(self, df):
        '''
        spotにインデックスをつける
        '''
        print('start spot_index')
        #df = self.df_spot_wikipedia
        index_spot = {i:spot for i,spot in enumerate(df['spot_name'])}
        spot_index = {spot:i for i,spot in enumerate(df['spot_name'])}

        with open(self.spot_index_path, 'wb') as f:
            pickle.dump(spot_index, f)

        with open(self.index_spot_path, 'wb') as f:
            pickle.dump(index_spot, f)
        print('end spot_index')
        return df

    def word_index(self, df):
        '''
        tf-idfに現れた単語の一覧をインデックスに直す
        '''
        print("word_index")
        tfidf_words = np.load(self.tfidf_word_path)
        print(tfidf_words)
        index_word = {i:word for i,word in enumerate(tfidf_words)}
        word_index = {word:i for i,word in enumerate(tfidf_words)}

        with open(self.index_word_path, 'wb') as f:
            pickle.dump(index_word, f)

        with open(self.word_index_path, 'wb') as f:
            pickle.dump(word_index, f)
        print('end word_index')
        return df

    def spot_spot_distance(self):
        df = pd.read_csv(self.df_experience_light_path)
        latlon = df[['latitude', 'longitude']].values
        nn = NearestNeighbors(n_neighbors=1000,)
        nn.fit(latlon)
        neighbors=nn.kneighbors(latlon)[1]
        spot_from = defaultdict(list)
        spot_to = defaultdict(list)

        for i,n in tqdm(enumerate(neighbors)):
            for j in n:
                from_lat, from_lon = df.loc[i,'latitude'], df.loc[i, 'longitude']
                to_lat, to_lon = df.loc[j, 'latitude'], df.loc[j, 'longitude']    
                dis = geodesic((from_lat, from_lon), (to_lat, to_lon)).km
                for km in [0.1, 0.2, 0.5, 1, 2, 5, 10]:
                    if dis<km:
                        spot_from[km].append(i)
                        spot_from[km].append(j)
                        spot_to[km].append(i)
                        spot_to[km].append(j)

        for km in [0.1, 0.2, 0.5, 1, 2, 5, 10]:
            path = os.path.join(self.data_graph_dir, f'spot_spot_{str(km)}.npy')
            edge_index = np.array([spot_from[km], spot_to[km]])
            np.save(path, edge_index)

    def spot_word(self):
        '''
        spotからwordへのedgeのインデックスを構築
        '''
        print('start spot_word')
        #df = self.df_spot_wikipedia_text
        tfidf_topk_index = np.load(self.tfidf_topk_index_path)
        edge_from = []
        edge_to = []
        for spot_ind, index in enumerate(tfidf_topk_index):
            for word_ind in index:
                edge_from.append(spot_ind)
                edge_to.append(word_ind)

        edge_index = np.array([edge_from, edge_to])
        np.save(self.spot_word_path, edge_index)
        print('end spot_word')
    

    def spot_category(self, df):
        '''
        spotからカテゴリーのedgeのインデックスを構築
        '''
        print('start spot category')
        spot_from = []
        category_to = []
        index = 0
        jenre_dict = {}
        for i,jenre in enumerate(df['jenre']):
            jenre = jenre.split(',')
            for j in jenre:
                if j not in jenre_dict:
                    jenre_dict[j]=index
                    index+=1
                target_index = jenre_dict[j]
                spot_from.append(i)
                category_to.append(target_index)
        jenre_index = jenre_dict
        index_jenre = {v:k for k,v in jenre_index.items()}
        spot_category = np.array([spot_from, category_to])
        with open(self.category_index_path, 'wb') as f:
            pickle.dump(jenre_index,f)

        with open(self.index_category_path, 'wb') as f:
            pickle.dump(index_jenre, f)
        np.save(self.spot_category_path, spot_category)
        print('end category')
        return df

    def city_index(self):
        print('start city index')
        with open(self.city_adj_path,'rb') as f:
            city_adj = pickle.load(f)
        city_index = {city:index for index, city in enumerate(city_adj.keys())}
        index_city = {index:city for index, city in enumerate(city_adj.keys())}

        with open(self.city_index_path,'wb') as f:
            pickle.dump(city_index,f)

        with open(self.index_city_path, "wb") as f:
            pickle.dump(index_city, f)
        print('end city index')


    def spot_city(self):
        '''
        spotからcityへのindexを構築
        '''
        special_pattern={
            "余市町（余市郡）":"余市町",
            "市貝町（芳賀郡）":"市貝町",
            "廿日市市": "廿日市市",
            "市川市": "市川市",
            "市原市": "市原市",
            "市川三郷町（西八代郡）": "市川三郷町",
            "上市町（中新川郡）": "上市町",
            "野々市市": "野々市市",
            "四日市市": "四日市市",
            "市川町（神崎郡）":"市川町",
            "下市町（吉野郡）": "下市町",
            }
        def city_process(city):
            orig_city=city
            if '（' in city:
                city= city[:city.find('（')]
            if '市' in city:
                city = city[:city.find('市')+1]
            if orig_city in special_pattern:
                city = special_pattern[orig_city]
            return city
        
        df = pd.read_csv(self.df_experience_path)
        df['city_processed']=df['city'].apply(city_process)
        with open(self.city_index_path, 'rb') as f:
            city_index = pickle.load(f)

        spot_from = []
        city_to = []
        for i, city in enumerate(df['city_processed']):
            spot_from.append(i)
            city_to.append(city_index[city])

        spot_city = np.array([spot_from, city_to])
        np.save(self.spot_city_path, spot_city)
        print('spot city done')
        df.to_csv(self.df_experience_path)
        return df

    def city_adj(self):
        url = 'https://uub.jp/cpf/rinsetsu.html'
        html = requests.get(url)
        soup = BeautifulSoup(html.content, 'lxml')
        adj_informations=soup.find_all('tr', class_="al bw")
        city_adj = {}
        for i in range(len(adj_informations)):
            adj_information = adj_informations[i]
            adj_from_pattern = re.compile(r'<td>(.*?)</td>')
            adj_to_pattern = re.compile(r'<nobr>(.*?)</nobr>')
            adj_from = re.search(adj_from_pattern, str(adj_information)).group(1)
            adj_to = adj_to_pattern.findall(str(adj_information))
            city_adj[adj_from] = adj_to
        with open(self.city_adj_path, 'wb') as f:
            pickle.dump(city_adj, f)
        print('city adj done')


    def city_pref(self):
        df = pd.read_csv(self.df_experience_path)
        edge =set()
        for city_ind, pref_ind in zip(df['city_label'], df['pref_label']):
            edge.add((city_ind, pref_ind))

        edges = np.array([[e[0] for e in edge],[e[1] for e in edge]])
        np.save(self.city_pref_path, edges)
        print('city pref done')

    def city_city(self):
        with open(self.city_adj_path, 'rb') as f:
            city_adj = pickle.load(f)

        with open(self.city_index_path,'rb') as f:
            city_index = pickle.load(f)

        adj_from = []
        adj_to=[]
        for k,v in city_adj.items():
            for c in v:
                if  city_index.get(k,False) and city_index.get(c,False):
                    print(k,c)
                    adj_from.append(city_index.get(k))
                    adj_to.append(city_index.get(c))

        edges = np.array([adj_from, adj_to])
        np.save(self.city_city_path, edges)
        print('city city done')

    def pref_index(self):
        df = pd.read_csv(self.df_experience_light_path)
        prefs = df['prefecture'].unique()
        pref_index = {pref:index for index, pref in enumerate(prefs)}
        index_pref = {index:pref for index, pref in enumerate(prefs)}
        with open(self.pref_index_path, 'wb') as f:
            pickle.dump(pref_index, f)

        with open(self.index_pref_path, 'wb') as f:
            pickle.dump(index_pref, f)

    def city_pref(self):
        df = pd.read_csv(self.df_experience_light_path)
        with open(self.city_index_path, 'rb') as f:
            city_index = pickle.load(f)

        with open(self.pref_index_path, 'rb') as f:
            pref_index = pickle.load(f)

        edge=set()
        for city, pref in zip(df['city_processed'], df['prefecture']):
            city_ind = city_index[city]
            pref_ind = pref_index[pref]
            edge.add((city_ind, pref_ind))

        edges = np.array([[e[0] for e in edge],[e[1] for e in edge]])
        np.save(self.city_pref_path, edges)
        print('city pref done')

    def pref_pref(self):
        with open(self.pref_index_path, 'rb') as f:
            pref_index = pickle.load(f)
        edge = set()
        for i in range(1,48):
            pref_from = pref_names[i]
            from_index = pref_index[pref_from]
            for j in adjacent(i):
                pref_to = pref_names[j]
                to_index = pref_index[pref_to]
                edge.add((from_index, to_index))

        edge_from = np.array([e[0] for e in edge])
        edge_to = np.array([e[1] for e in edge])
        edges = np.array([edge_from, edge_to])
        np.save(self.pref_pref_path, edges)
        print('pref pref done')


     

    def spot_pref(self, df):
        '''
        spotからprefへのindexを構築
        '''
        le = LabelEncoder()
        df['pref_label'] = le.fit_transform(df['prefecture'])
        pref_index = {i:pref for i,pref in enumerate(le.classes_)}
        with open(self.pref_index_path, 'wb') as f:
            pickle.dump(pref_index, f)

        edge_from, edge_to = [],[]
        for spot_ind, pref_ind in enumerate(df['pref_label'].values):
            edge_from.append(int(spot_ind))
            edge_to.append(int(pref_ind))

        spot_pref = np.array([edge_from, edge_to])
        np.save(self.spot_pref_path, spot_pref)
        print('spot_pref done')
        return df

    def spot_spot(self, df):
        edge_from = []
        edge_to = []
        for pref in tqdm(df['都道府県'].unique()):
            df_tmp = df[df['都道府県']==pref].reset_index(drop=False)
            for i in range(len(df_tmp)):
                for j in range(i+1, len(df_tmp)):
                    lat1, lng1 = df_tmp.loc[i, 'latitude'], df_tmp.loc[i, 'longitude']
                    lat2, lng2 = df_tmp.loc[i, 'latitude'], df_tmp.loc[i, 'longitude']
                    dist = geodesic((lat1,lng1),(lat2,lng2)).km
                    if dist<5:
                        index1 = df_tmp.loc[i, 'index']
                        index2 = df_tmp.loc[j, 'index']
                        edge_from.append(index1)
                        edge_to.append(index2)
            print(len(edge_from))
        spot_spot = np.array([edge_from, edge_to])
        #print(spot_spot)
        np.save(self.spot_spot_path, spot_spot)
        return df

    def word_popularity(self):
        words = np.load(self.tfidf_word_path)
        pageviews=[]
        for word in tqdm(words):
            try:
                pageview = pageviewapi.period.sum_last('ja.wikipedia', word, 
                        last=365,access="all-access", agent="all-agents")
            except pageviewapi.client.ZeroOrDataNotLoadedException:
                pageview = 0
            pageviews.append(pageview)
            time.sleep(0.5)
        print(len(pageviews))
        np.save(self.word_popularity_path, np.array(pageviews))

    def city_popularity(self):
        with open(self.city_index_path, 'rb') as f:
            city_index = pickle.load(f)

        city_popularity = []
        for city, index in city_index.items():
            try:
                pageview = pageviewapi.period.sum_last('ja.wikipedia', city,
                        last = 365, access = 'all-access', agent = 'all-agents')
            except pageviewapi.client.ZeroOrDataNotLoadedException:
                pageview = 0
            city_popularity.append(pageview)
            print(city, pageview)
            time.sleep(0.5)
        np.save(self.city_popularity_path, np.array(city_popularity))

    def pref_popularity(self):
        with open(self.pref_index_path, 'rb') as f:
            pref_index = pickle.load(f)
        session = HTMLSession()
        r = session.get('https://uub.jp/pdr/s/m.html')
        soup = BeautifulSoup(r.html.html, 'lxml')
        prefs=soup.select('tr.vt tr td')
        prefs = [pref.text for pref in prefs][1::3]
        pref_scores = soup.select('tr td.ar')[:47]
        pref_scores = [float(s.text) for s in pref_scores]
        pref_y = [0] * 47
        for pref, pref_score in zip(prefs, pref_scores):
            pref_y[pref_index[pref]] = pref_score

        np.save(self.pref_popularity_path, pref_y)



    def construct_graph(self, df):
        df = self.spot_index(df)
        df = self.word_index(df)
        df = self.spot_word(df)
        df = self.spot_city(df)
        df = self.spot_category(df)
        self.word_attr()
        self.spot_spot(df)
        df.to_csv(self.df_experience_path)
        return df

    def construct_after_tokenize(self, df):
        df = self.word_index(df)
        df = self.spot_word(df)
        self.word_attr()
        #self.word_emb_clip()
        df.to_csv(self.df_experience_path)
        return df
            
    def word_attr(self, th=False):
        print('start word attr')
        model = gensim.models.Word2Vec.load('./data/ja/ja.bin')
        words = np.load(self.tfidf_word_path)
        if th==True:
            words = np.load(self.tfidf_word_th_path)
        emb_dim = len(model['日本'])
        word_embs = []
        for word in tqdm(words):
            #print(word)
            try:
                emb = model[word]
            except KeyError:
                emb = np.zeros(emb_dim)

            word_embs.append(emb)

        word_embs = np.array(word_embs)
        if th:
            np.save(self.word_embs_th_path, word_embs)
        else:
            np.save(self.word_embs_path, word_embs)
        print('end word attr')

    def word_attr_wiki(self):
        print('start word attr')
        model = gensim.models.Word2Vec.load('./data/ja/ja.bin')
        words = np.load(self.tfidf_word_path)
        emb_dim = len(model['日本'])
        word_embs = []
        for word in tqdm(words):
            #print(word)
            try:
                emb = model[word]
            except KeyError:
                emb = np.zeros(emb_dim)

            word_embs.append(emb)

        word_embs = np.array(word_embs)
        np.save(self.word_embs_wiki_path, word_embs)
        print('end word attr')


    def word_attr_finetune(self):
        print('start word attr finetune')
        model = gensim.models.Word2Vec.load(os.path.join(self.data_graph_dir,'ja_review.model'))
        words = np.load(self.tfidf_word_path)
        emb_dim = len(model['日本'])
        word_embs = []
        for word in tqdm(words):
            #print(word)
            try:
                emb = model[word]
            except KeyError:
                emb = np.zeros(emb_dim)

            word_embs.append(emb)

        word_embs = np.array(word_embs)
        np.save(self.word_embs_finetune_path, word_embs)
        print('end word attr')

    def word_attr_wiki(self):
        print('start word attr')
        model = gensim.models.Word2Vec.load('./data/ja/ja.bin')
        words = np.load(self.tfidf_word_path)
        emb_dim = len(model['日本'])
        word_embs = []
        for word in tqdm(words):
            #print(word)
            try:
                emb = model[word]
            except KeyError:
                emb = np.zeros(emb_dim)

            word_embs.append(emb)

        word_embs = np.array(word_embs)
        np.save(self.word_embs_wiki_path, word_embs)
        print('end word attr')


    def word_attr_ensemble(self):
        print('start word attr finetune')
        model_finetune = gensim.models.Word2Vec.load(os.path.join(self.data_graph_dir,'ja_review.model'))
        model_wiki = gensim.models.Word2Vec.load('./data/ja/ja.bin')
        words = np.load(self.tfidf_word_path)
        emb_dim_finetune = len(model_finetune['日本'])
        emb_dim_wiki = len(model_wiki['日本'])
        word_embs = []
        for word in tqdm(words):
            #print(word)
            try:
                emb_finetune = model_finetune[word]
            except KeyError:
                emb_finetune = np.zeros(emb_dim_finetune)
            try:
                emb_wiki = model_wiki[word]
            except KeyError:
                emb_wiki = np.zeros(emb_dim_wiki)
            word_embs.append(np.concatenate([emb_finetune, emb_wiki]))

        word_embs = np.array(word_embs)
        np.save(self.word_embs_ensemble_path, word_embs)
        print('end word attr')

    def word_emb_clip(self):
        print('start word emb clip')
        device = "cuda:7" if torch.cuda.is_available() else "cpu"
        word_emb_all =[]
        words = np.load(self.tfidf_word_path)
        model, preprocess = ja_clip.load("rinna/japanese-cloob-vit-b-16", device=device)
        tokenizer = ja_clip.load_tokenizer()
        for i in range(0,len(words),128):
            encodings = ja_clip.tokenize(
                texts=list(words[i:i+128]),
                max_seq_len=20,
                device=device,
                tokenizer=tokenizer, # this is optional. if you don't pass, load tokenizer each time
            )
            word_emb = model.get_text_features(**encodings)
            word_emb_all.append(word_emb.cpu().detach().numpy())
        word_emb_all = np.concatenate(word_emb_all)
        np.save(self.word_emb_clip_path, word_emb_all)
        print('end word emb clip')

    def spot_image_feature(self, df, pretrain=False, reduce_dim=False, dim=512, model_name='ResNet'):
        if model_name=='ResNet':
            model = torchvision.models.resnet18(pretrained=True)
        elif model_name=='VGG':
            model = torchvision.models.vgg16(pretrained=True)

        if pretrain==True:
            model.load_state_dict(torch.load('./data/best_model_image.bin'))
        model.fc = nn.Identity()
        model.eval()
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        device='cuda'
        features = []
        model.to(device)
        for spot_name in tqdm(df['spot_name']):
            spot_name = spot_name.replace('/','')
            img_path = os.path.join(self.jalan_image_dir, f'{spot_name}_0.jpg')
            img = Image.open(img_path).convert('RGB')
            img = preprocess(img).unsqueeze(0)
            img.to(device)
            feature = model(img)
            #print(spot_name)
            features.append(feature.to('cpu').detach().numpy().copy().reshape(-1))
        
        features = np.array(features)
        if reduce_dim == True:
            pca = PCA(n_components=dim)
            vec = pca.fit_transform(features)
            print(vec.shape)
        img_emb_path = os.path.join(self.data_graph_dir, f'spot_img_emb_{model_name}.npy')
        np.save(img_emb_path, features)

    def spot_aerial_feature(self, df, pretrain=False, reduce_dim=False, dim=512, model_name='ResNet'):
        if model_name=='ResNet':
            model = torchvision.models.resnet18(pretrained=True)
        elif model_name=='VGG':
            model = torchvision.models.vgg16(pretrained=True)

        if pretrain==True:
            model.load_state_dict(torch.load('./data/best_model_image.bin'))
        model.fc = nn.Identity()
        model.eval()
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        device='cuda:0'
        features = []
        model.to(device)
        count=0
        for spot_name in tqdm(df['spot_name']):
            spot_name = spot_name.replace('/','')
            img_path = os.path.join('./data/aerial', f'{spot_name}.jpg')
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                img = preprocess(img).unsqueeze(0)
                img=img.to(device)
                feature = model(img)
                #print(spot_name)
                features.append(feature.to('cpu').detach().numpy().copy().reshape(-1))
            else:
                count+=1
                #print(spot_name)
                feature = np.random.rand(512)*0.1
                features.append(feature)
        print(count)
        features = np.array(features)
        if reduce_dim == True:
            pca = PCA(n_components=dim)
            vec = pca.fit_transform(features)
            print(vec.shape)
        aerial_emb_path = os.path.join(self.data_graph_dir, f'aerial_img_emb_{model_name}.npy')
        np.save(aerial_emb_path, features)

    def category_image_feature(self, pretrain=False,reduce_dim=False, dim=256):
        model = torchvision.models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features

        if pretrain==True:
            model.load_state_dict(torch.load('./data/best_model_image.bin'))
        model.fc = nn.Identity()
        model.eval()
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        features = []
        with open(self.category_index_path,'rb') as f:
            category_index= pickle.load(f)

        for category in tqdm(category_index.values()):
            img_path = os.path.join(self.category_image_dir, f'{category}.jpg')
            if os.path.exists(img_path):
                print(category)
                img = Image.open(img_path).convert('RGB')
                img = preprocess(img).unsqueeze(0)
                feature = model(img)
                features.append(feature.to('cpu').detach().numpy().copy().reshape(-1))
            else:
                feature = np.random.rand(num_ftrs)*0.01
                print(feature)
                features.append(feature)
        
        features = np.array(features)
        print(features.shape)
        if reduce_dim == True:
            pca = PCA(n_components=dim)
            vec = pca.fit_transform(features)
            print(vec.shape)

        np.save(self.category_img_emb_path, features)


    def spot_image_feature_multi(self, pretrain=False, reduce_dim=False, dim=512, model_name='ResNet'):
        df = pd.read_csv(self.df_experience_path)
        if model_name=='ResNet':
            model = torchvision.models.resnet18(pretrained=True)
        elif model_name=='VGG':
            model = torchvision.models.vgg16(pretrained=True)
        elif model_name=='ViT':
            model = ViT('B_16_imagenet1k', pretrained=True)
            print(model)

        if pretrain==True:
            model.load_state_dict(torch.load('./data/best_model_image.bin'))
        if model!='ViT':
            model.fc = nn.Identity()
        model.eval()
        if model_name!='ViT':
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
        else:
            preprocess = transforms.Compose([
                transforms.Resize((384, 384)), 
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ])
        features = []
        for spot_name in tqdm(df['spot_name']):
            spot_name = spot_name.replace('/','')
            feature = []
            img_paths =[]
            img_num=5
            for i in range(img_num):
                img_path = os.path.join(self.jalan_image_dir, f'{spot_name}_{i}.jpg')
                if os.path.exists(img_path):
                    img_paths.append(img_path)
            
            if len(img_paths)<img_num:
                print(spot_name, len(img_paths))
                img_paths.extend([img_paths[0]]*(img_num-len(img_paths)))
            
            for img_path in img_paths:
                img = Image.open(img_path).convert('RGB')
                img = preprocess(img).unsqueeze(0)
                feature.append(model(img).to('cpu').detach().numpy().copy().reshape(-1))
            #print(spot_name)
            features.append(np.concatenate(feature))
        
        features = np.array(features)
        print(features.shape)
        if reduce_dim == True:
            pca = PCA(n_components=dim)
            features = pca.fit_transform(features)
            print(features.shape)
        img_emb_path = os.path.join(self.data_graph_dir, f'spot_img_emb_multi_{model_name}.npy')
        np.save(img_emb_path, features)

    def spot_image_feature_clip(self, df, reduce_dim=False, dim=512):
        print('image_feature_clip start')
        device = "cuda:7" if torch.cuda.is_available() else "cpu"
        model, preprocess = ja_clip.load("rinna/japanese-cloob-vit-b-16", device=device)
        tokenizer = ja_clip.load_tokenizer()

        features = []
        for spot_name in tqdm(df['spot_name']):
            img_path = os.path.join(self.jalan_image_dir, f'{spot_name}.jpg')
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                img = preprocess(img).unsqueeze(0).to(device)
                feature = model.get_image_features(img)
            else:
                feature
            features.append(feature.to('cpu').detach().numpy().copy().reshape(-1))
        
        features = np.array(features)
        np.save(self.spot_img_emb_clip_path, features)
        print('image_feature_clip end')

    def pref_attr(self):
        data_dir='./data/pref'
        csvs = os.listdir('./data/pref')
        dfs = [pd.read_csv(os.path.join(data_dir,csv), encoding='shift-JIS') for csv in csvs]
        df = dfs[0]
        for df_ in dfs[1:]:
            df = pd.merge(df,df_, on='地域')

        df = df.loc[:,~df.columns.str.contains('項目')]
        df = df.loc[:,~df.columns.str.contains('調査年')]
        df = df.set_index('地域')
        path = Path()
        with open(path.index_pref_path,'rb') as f:
            index_pref = pickle.load(f)
        df = df.select_dtypes(exclude='object')
        df = df.iloc[:,[0,19,25,26,29,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,56,65,66]]
        df.to_csv('./data/pref.csv')

        ss = StandardScaler()
        X=ss.fit_transform(df.values)
        features = np.zeros_like(X)
        for ind, pref in index_pref.items():
            ind_org = list(df.index).index(pref)
            features[ind] = X[ind_org]

        np.save(self.pref_attr_path, features)
        print('pref attr done')

    def city_attr(self):
        def get_last(text):
            text = text.split(' ')[-1]
            return text
        data_dir='./data/city'
        csvs = os.listdir('./data/city')
        dfs = [pd.read_csv(os.path.join(data_dir,csv), encoding='shift-JIS') for csv in csvs]
        df = dfs[0]
        for df_ in dfs[1:]:
            df = pd.merge(df,df_, on='地域')

        df = df.loc[:,~df.columns.str.contains('項目')]
        df = df.loc[:,~df.columns.str.contains('調査年')]
        df['地域'] = df['地域'].apply(get_last)
        df = df.set_index('地域')
        df.replace('-',None)
        df.fillna(df.mean())
        path = Path()
        with open(path.city_index_path,'rb') as f:
            city_index = pickle.load(f)
        df = df.select_dtypes(exclude='object')
        df.to_csv('./data/city.csv')
        df.head()
        ss = StandardScaler()
        X=ss.fit_transform(df.values)
        features = np.zeros((len(city_index), X.shape[1]))

        for ind, city in city_index.items():
            if city in df.index:
                ind_org = list(df.index).index(city)
                features[ind] = X[ind_org]

        np.save(path.city_attr_path, features)
        print('city attr done')
        return features


class PreProcessing(Path):
    def __init__(self):
        super().__init__()
        self.Makedf = Makedf()
        self.Tokenize = Tokenize()
        self.TFIDF = TFIDF()
        self.ConstructGraph = ConstructGraph()

    def preprocessing(self):
        df = self.df_spot_wikipedia
        df = self.Makedf.make_experience_df(df)
        print('len experience df', len(df))
        df = self.Tokenize.tokenize_text(df, column ='wiki_text')
        df = self.TFIDF.tfidf_topk(df,k=20)
        df = self.ConstructGraph.construct_graph(df)
        df.to_csv(self.df_experience_path)

    def tokenize_and_tfidf(self, column='jalan_review', hinshi=['名詞','形容詞'], k=10):
        df = pd.read_csv(self.df_experience_path)
        df = self.Tokenize.tokenize_text(df, column=column, hinshi=hinshi)
        df = self.TFIDF.tfidf_topk(df, k=k)
        df = self.ConstructGraph.construct_after_tokenize(df)
        df.to_csv(self.df_experience_path)

    def tfidf(self,k=10):
        df = pd.read_csv(self.df_experience_path)
        df = self.TFIDF.tfidf_topk(df, k=k)
        df = self.ConstructGraph.construct_after_tokenize(df)
        df.to_csv(self.df_experience_path)

    def tfidf_th(self,threshold=0.1):
        df = pd.read_csv(self.df_experience_path)
        df = self.TFIDF.tfidf_threshold(df, threshold=threshold)
        self.ConstructGraph.word_attr()
        


if __name__=='__main__':
    path = Path()
    #df = pd.read_csv(path.df_experience_light_path)
    #cg = ConstructGraph()
    #cg.spot_aerial_feature(df)
    #exit()
    #cg.spot_pref(df)
    #cg.spot_image_feature_multi('ResNet')
    #cg.word_attr()
    #cg.word_attr_wiki()
    #cg.word_attr_ensemble()
    #cg.spot_image_feature_multi(model_name='ViT')
    #cg.spot_image_feature_multi(model_name='VGG')
    #exit()
    #cg.word_attr_ensemble()
    '''
    md = Makedf()
    df=md.join_review(df)
    df=md.join_text(df)
    '''
    '''
    path = Path()
    df = pd.read_csv(path.df_experience_path)
    tokenize = Tokenize()
    df = tokenize.clean_text(df)
    tfidf = TFIDF()
    df=df.fillna({'tokenized_text_clean':'観光地'})
    df = tfidf.tfidf_threshold(df, column='tokenized_text_clean', threshold=0.1)
    cg = ConstructGraph()
    cg.word_attr(th=True)
    '''
    #cg = ConstructGraph()
    #cg.spot_spot_distance()
    #md = Makedf()
    #md.get_stratified_fold_split(df)
    md = Makedf()
    path = Path()
    df = pd.read_csv(path.df_experience_path)
    df=md.join_review(df)
    df=md.join_text(df)
    tokenize = Tokenize()
    df = tokenize.tokenize_text(df)
    df=tokenize.clean_text(df)
    df.to_csv(path.df_experience_spare_path)
    tfidf = TFIDF()
    df=tfidf.tfidf_topk_spare(df, column='tokenized_text_clean', k=15)
    #cg = ConstructGraph()
    #cg.spot_word()
    #cg.word_attr_finetune()
    
    #df.to_csv(path.df_experience_path)
    #df = pd.read_csv(path.df_experience_path)
    #cg.word_attr_wiki()
    #df=md.get_stratified_fold_split(df)
    #cg.spot_image_feature(df)
    #cg.word_index(df)
    #cg.spot_word(df)
    #df.to_csv(path.df_experience_path)
    
    
    #PP = PreProcessing()
    #PP.Makedf.parse_geocode_pickle()
    #path = Path()
    #df = pd.read_csv(path.df_experience_path)
    #PP.ConstructGraph.word_attr_finetune()
    #PP.tfidf_th(threshold=0.05)
    #df = pd.read_csv(PP.df_experience_path)
    #PP.TFIDF.tfidf_topk(df)
    #PP.ConstructGraph.spot_image_feature_multi(pretrain=False)
    #PP.tokenize_and_tfidf()
    #df = pd.read_csv(PP.df_experience_path)

    #PP.ConstructGraph.spot_image_feature_clip(df)
    #PP.tokenize_and_tfidf()
    #df = pd.read_csv(PP.df_experience_path)
    #df=PP.ConstructGraph.spot_category(df)
    #df=PP.ConstructGraph.spot_city(df)
    #df=PP.ConstructGraph.spot_pref(df)
    #df.to_csv(PP.df_experience_path)
    #df = pd.read_csv(PP.df_experience_path)
    #PP.ConstructGraph.spot_spot(df)
    #PP.spot_image_feature()
    #PP.make_experience_df()
    #PP.get_stratified_fold_split()
    #PP.tokenize_text()
    #PP.tfidf_topk()
    #PP.spot_index()
    #PP.word_index()
    #PP.spot_word()
    #PP.word_attr()