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

trained_pickle = open ("trained_pickle", "rb")
train = pickle.load(trained_pickle)

lin = LinearRegression()
rf = RandomForestClassifier()
kernel = 1.0 * RBF(1.0)
gpc = GaussianProcessClassifier(kernel=kernel, random_state=0)

def train_model(df, model):
    model.fit(df[['Tweet Weighted Topic Score', 'Bio Weighted Topic Score', 'Tweet Sentiment Score', 'Bio Sentiment Score',
                  'ZS Weighted Score', 'Bio ZS "Startup"', 'Bio ZS "Startup Founder"', 
                  'Word Match Score', 'Self-Link Similarity']].values, df['Output'])
    
def train_gaussian_model(df, model):
    model.fit(df[['Tweet Weighted Topic Score', 'Bio Weighted Topic Score', 'Tweet Sentiment Score', 'Bio Sentiment Score',
                  'ZS Weighted Score', 'Bio ZS "Startup"', 'Bio ZS "Startup Founder"', 
                  'Word Match Score', 'Self-Link Similarity']].values, df['Output'])

train_model(train, lin)
train_model(train, rf)
train_gaussian_model(train, gpc)

file_lin = open('lin', 'wb')
pickle.dump(lin, file_lin)
file_lin.close()

file_rf = open('rf', 'wb')
pickle.dump(rf, file_rf)
file_rf.close()

file_gpc = open('gpc', 'wb')
pickle.dump(gpc, file_gpc)
file_gpc.close()