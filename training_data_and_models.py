import snscrape.modules.twitter as sntwitter
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from transformers import pipeline
from scipy.special import expit, softmax
import pickle
import Levenshtein

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from datetime import date
from datetime import timedelta


train = pd.read_csv('train1.csv').fillna('')[['Tweet', 'User Bio', 'Username','Link', 'Output']]

MODEL_topic = f"cardiffnlp/tweet-topic-21-multi"
tokenizer_topic = AutoTokenizer.from_pretrained(MODEL_topic)
model_topic = AutoModelForSequenceClassification.from_pretrained(MODEL_topic)
class_mapping = model_topic.config.id2label
file_topic = open('model_topic', 'wb')
pickle.dump(model_topic, file_topic)
file_topic.close()

MODEL_sentiment = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer_sentiment = AutoTokenizer.from_pretrained(MODEL_sentiment)
config_sentiment = AutoConfig.from_pretrained(MODEL_sentiment)
model_sentiment = AutoModelForSequenceClassification.from_pretrained(MODEL_sentiment)
file_sentiment = open('model_sentiment', 'wb')
pickle.dump(model_sentiment, file_sentiment)
file_sentiment.close()

zeroshot_classifier = pipeline("zero-shot-classification")
file_zs = open('model_zs', 'wb')
pickle.dump(zeroshot_classifier, file_zs)
file_zs.close()

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def generate_topic_scores(text):
    tokens = tokenizer_topic(text, return_tensors='pt')
    output = model_topic(**tokens)
    scores = expit(output[0][0].detach().numpy())
    return [scores[1], scores[10], scores[15]]

def generate_sentiment_score(text):
    text = preprocess(text)
    encoded_input = tokenizer_sentiment(text, return_tensors='pt')
    output = model_sentiment(**encoded_input)
    scores = softmax(output[0][0].detach().numpy())
    return [scores[2], scores[1]]

def generate_zeroshot_classification(text, label):
    if text=='': return 0.0
    return zeroshot_classifier(text, candidate_labels=[label])['scores'][0]

def generate_zeroshot_weighted_score(df):
    l = []
    for i in range(len(df['ZS "Startup"'])):
        a = [df['ZS "Startup"'][i], df['ZS "Announcement"'][i], df['ZS "Innovation"'][i]]
        a.sort()
        l.append((a[2]+a[1])/2)
    df['ZS Weighted Score'] = l
    return df


word_match_bag = {'startup': 2.5, 'launch':2, 'company':1, 'found':0.5, 'raise':2, '$':1, 'accelerator':2, 'code':1, 'hiring':2, 'release':1, 'announce':0.25, 'batch':1, 'transform':0.5,
                  'improv':0.5, 'mission':0.5, 'seed':1.5, 'incubator':1.5, 'workflow':0.5, 'backed by':2, 'the first':2, 'enable':0.5, 'learn more':0.5, 'business':0.5,
                  'NFT':-1, 'metaverse':-0.5, 'available':-0.1, 'DAO':-0.75}

word_match_bag_sans_startup = {'launch':2, 'company':1, 'found':0.5, 'raise':2, '$':1, 'accelerator':2, 'code':1, 'hiring':2, 'release':1, 'announce':0.25, 'batch':1, 'transform':0.5,
                  'improv':0.5, 'mission':0.5, 'seed':1.5, 'incubator':1.5, 'workflow':0.5, 'backed by':2, 'the first':2, 'enable':0.5, 'learn more':0.5, 'business':0.5,
                  'NFT':-1, 'metaverse':-0.5, 'available':-0.1, 'DAO':-0.75}

word_match_bag_sans_raise = {'startup': 2.5, 'launch':2, 'found':0.5, 'company':1, '$':1, 'accelerator':2, 'code':1, 'hiring':2, 'release':1, 'announce':0.25, 'batch':1, 'transform':0.5,
                  'improv':0.5, 'mission':0.5, 'seed':1.5, 'incubator':1.5, 'workflow':0.5, 'backed by':2, 'the first':2, 'enable':0.5, 'learn more':0.5, 'business':0.5,
                  'NFT':-1, 'metaverse':-0.5, 'available':-0.1, 'DAO':-0.75}

def generate_word_match_score(text, words_dict):
    score = 0
    for word in words_dict.keys():
        if text.lower().find(word)>-1: score +=words_dict[word]
    return score

def generate_features_and_prune(df):
    df['Word Match Score']=df['Tweet'].map(lambda x: generate_word_match_score(x, word_match_bag))
    
    df=df[(df['Word Match Score']>0.01)].reset_index(drop=True)
    
    tweet_topic_scores = df['Tweet'].map(generate_topic_scores)
    df['Tweet Weighted Topic Score']=(tweet_topic_scores.map(lambda x: x[0])*3+tweet_topic_scores.map(lambda x: x[1])+tweet_topic_scores.map(lambda x: x[2]))/5

    df=df[(df['Tweet Weighted Topic Score']>0.25)].reset_index(drop=True)
    
    bio_topic_scores = df['User Bio'].map(generate_topic_scores)
    df['Bio Weighted Topic Score']=(bio_topic_scores.map(lambda x: x[0])*2+bio_topic_scores.map(lambda x: x[1])+bio_topic_scores.map(lambda x: x[2]))/4
    df=df[(df['Bio Weighted Topic Score']>0.10)].reset_index(drop=True)
    
    tweet_sentiment_scores = df['Tweet'].map(generate_sentiment_score)
    df['Tweet Sentiment Score']=tweet_sentiment_scores.map(lambda x: (x[0]*3+x[1])/4)
    df=df[(df['Tweet Sentiment Score']>0.25)].reset_index(drop=True)
    
    print('25% done with generating columns.')
    bio_sentiment_scores = df['User Bio'].map(generate_sentiment_score)
    df['Bio Sentiment Score']=bio_sentiment_scores.map(lambda x: (x[0]*3+x[1])/4)
    
    df['Self-Link Similarity']=[Levenshtein.ratio(df['Link'][i],df['Username'][i])for i in range(len(df['Link']))] 
        
    df['ZS "Startup"']=[generate_zeroshot_classification(x, 'startup') for x in df['Tweet']]
    df=df[(df['ZS "Startup"']>0.10)].reset_index(drop=True)
    
    print('50% done with generating columns.')
    
    df['ZS "Announcement"']=[generate_zeroshot_classification(x, 'announcement') for x in df['Tweet']]
    df=df[(df['ZS "Announcement"']>0.10)].reset_index(drop=True)
    
    df['ZS "Innovation"']=[generate_zeroshot_classification(x, 'announcement') for x in df['Tweet']]
    df=df[(df['ZS "Innovation"']>0.10)].reset_index(drop=True)
    
    print('75% done with generating columns.')
    
    df = generate_zeroshot_weighted_score(df)
    
    df['Bio ZS "Startup"']=[generate_zeroshot_classification(x, 'startup') for x in df['User Bio']]
    
    print('90% done with generating columns.')
    df['Bio ZS "Startup Founder"']=[generate_zeroshot_classification(x, 'startup founder') for x in df['User Bio']]
    
    return df

def train_model(df, model):
    model.fit(df[['Tweet Weighted Topic Score', 'Bio Weighted Topic Score', 'Tweet Sentiment Score', 'Bio Sentiment Score',
                  'ZS Weighted Score', 'Bio ZS "Startup"', 'Bio ZS "Startup Founder"', 
                  'Word Match Score', 'Self-Link Similarity']].values, df['Output'])
    
def train_gaussian_model(df, model):
    model.fit(df[['Tweet Weighted Topic Score', 'Bio Weighted Topic Score', 'Tweet Sentiment Score', 'Bio Sentiment Score',
                  'ZS Weighted Score', 'Bio ZS "Startup"', 'Bio ZS "Startup Founder"', 
                  'Word Match Score', 'Self-Link Similarity']].values, df['Output'])


