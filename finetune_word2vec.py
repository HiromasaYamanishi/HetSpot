from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
import gensim
from gensim.models import Word2Vec, KeyedVectors
import pandas as pd
from tqdm import tqdm
import MeCab
import numpy as np
from janome.charfilter import RegexReplaceCharFilter, UnicodeNormalizeCharFilter

def tokenize(text):
    t = MeCab.Tagger("")
    t.parse("")
    m = t.parseToNode(text)
    tokens = []
    while m:
        tokenData = m.feature.split(",")
        if m.surface not in '、。！':
            tokens.append(m.surface)
        m = m.next
    tokens.pop(0)
    if len(tokens)>0:
        tokens.pop(-1)
    return tokens
if __name__ == '__main__':
    df = pd.read_csv('./data/review/review_all.csv')
    df_exp = pd.read_csv('./data/spot/experience_light.csv')
    not_test = df_exp[df_exp['valid']!=1]['spot_name']
    mask = np.load(path.valid_idx_path)
    df = df.loc[df['spot'].isin(not_test),:]
    char_filters = [UnicodeNormalizeCharFilter(),
                    RegexReplaceCharFilter('[#!:;<>{}・`.,()-=$/_\d\'"\[\]\|~]+', ' ')]
    
    analyzer = Analyzer(char_filters = char_filters)
    sentences = df['review']
    sentences_tokenized = [tokenize(s) for s in tqdm(sentences)]
    model = gensim.models.Word2Vec.load('./data/ja/ja.bin')
    model.min_count=3
    model.build_vocab(sentences_tokenized, update=True)
    total_examples = model.corpus_count
    model.train(sentences_tokenized, total_examples = total_examples, epochs = 500)
    model.save('./data/graph/ja_review.model')