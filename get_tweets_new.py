import snscrape.modules.twitter as sntwitter
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from transformers import pipeline
from scipy.special import expit, softmax
import pickle
import Levenshtein

from datetime import date
from datetime import timedelta

model_topic_pickle = open ("model_topic", "rb")
model_topic = pickle.load(model_topic_pickle)
model_sentiment_pickle = open ("model_sentiment", "rb")
model_sentiment = pickle.load(model_sentiment_pickle)
zeroshot_classifier_pickle = open ('model_zs', 'rb')
zeroshot_classifier = pickle.load(zeroshot_classifier_pickle)

MODEL_topic = "cardiffnlp/tweet-topic-21-multi"
tokenizer_topic = AutoTokenizer.from_pretrained(MODEL_topic)
MODEL_sentiment = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer_sentiment = AutoTokenizer.from_pretrained(MODEL_sentiment)
config_sentiment = AutoConfig.from_pretrained(MODEL_sentiment)
tokenizer_zs = AutoTokenizer.from_pretrained("typeform/distilbert-base-uncased-mnli")


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

    tweet_sentiment_scores = df['Tweet'].map(generate_sentiment_score)
    df['Tweet Sentiment Score']=tweet_sentiment_scores.map(lambda x: (x[0]*3+x[1])/4)
    df=df[(df['Tweet Sentiment Score']>0.25)].reset_index(drop=True)

    bio_sentiment_scores = df['User Bio'].map(generate_sentiment_score)
    df['Bio Sentiment Score']=bio_sentiment_scores.map(lambda x: (x[0]*3+x[1])/4)

    df['Bio ZS "Startup Founder"']=[generate_zeroshot_classification(x, 'startup founder') for x in df['User Bio']]

    return df

def twitter_link_string_processor(text):
    if text=="None": return
    return text[14:].partition(",")[0][1:-1]

def grab_tweets(days_since=2, number = 10000):
    query = """
            (we AND ("excited to" OR "thrilled to" OR "honored to" OR "happy to") AND (build OR release OR announce OR launch OR share OR deliver OR feature OR provide OR improve OR create OR help))
            
            
            since:{} until:{}
            """.format(date.today() - timedelta(days = days_since+1), date.today() - timedelta(days = days_since))

    tweets_list = []
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):  
        if i>number:
            break
        tweets_list.append([tweet.url, tweet.rawContent, tweet.user.username, tweet.user.rawDescription, tweet.user.followersCount,
                            tweet.mentionedUsers, twitter_link_string_processor(str(tweet.user.link))])

    return pd.DataFrame(tweets_list, columns=['URL', 'Tweet', 'Username', 'User Bio', 'Followers', 'Tags', 'Link']).fillna('')

def grab_tweets_2(days_since=2, number = 10000):
    query = """
            ("we raised" OR "we just raised" OR "we have raised") AND round
            
            
            since:{} until:{}
            """.format(date.today() - timedelta(days = days_since+1), date.today() - timedelta(days = days_since))

    tweets_list = []
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):  
        if i>number:
            break
        tweets_list.append([tweet.url, tweet.rawContent, tweet.user.username, tweet.user.rawDescription, tweet.user.followersCount,
                            tweet.mentionedUsers, twitter_link_string_processor(str(tweet.user.link))])

    return pd.DataFrame(tweets_list, columns=['URL', 'Tweet', 'Username', 'User Bio', 'Followers', 'Tags', 'Link']).fillna('')

def grab_tweets_3(days_since=2, number = 10000):
    query = """
            ("my startup" OR "my new startup" OR "our startup" OR "our new startup")
            
            
            since:{} until:{}
            """.format(date.today() - timedelta(days = days_since+1), date.today() - timedelta(days = days_since))

    tweets_list = []
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):  
        if i>number:
            break
        tweets_list.append([tweet.url, tweet.rawContent, tweet.user.username, tweet.user.rawDescription, tweet.user.followersCount,
                            tweet.mentionedUsers, twitter_link_string_processor(str(tweet.user.link))])

    return pd.DataFrame(tweets_list, columns=['URL', 'Tweet', 'Username', 'User Bio', 'Followers', 'Tags', 'Link']).fillna('')

test = grab_tweets(2)

def flag_crypto(text):
    words = {'web3', 'token'}        
    for word in words:
        if text.lower().find(word)>-1: return 1
    return 0


def remove_crypto(df):
    return df[df['Tweet'].map(flag_crypto)==0]

test= remove_crypto(test)

test = generate_features_and_prune(test)


scores = []
for i in range(0,test.shape[0]):
    scores.append(test['Word Match Score'][i]*0.8 + test['Tweet Weighted Topic Score'][i] +
             test['Tweet Sentiment Score'][i]*0.4 + 
             test['Bio Sentiment Score'][i]*0.3 + test['Bio ZS "Startup Founder"'][i]*0.75)

test['Human Score']=scores

output = test.sort_values(by=['Human Score'], ascending=False)[0:30][['URL', 'Tweet']]

output_2 = grab_tweets_2()[['URL', 'Tweet']]
test_3 = grab_tweets_3()
test_3['Word Match Score']=test_3['Tweet'].map(lambda x: generate_word_match_score(x, word_match_bag_sans_startup))
output_3 = test_3[test_3['Word Match Score']>1][['URL', 'Tweet']]

output.to_csv('{} search 1.csv'.format(date.today()))
output_2.to_csv('{} search 2.csv'.format(date.today()))
output_3.to_csv('{} search 3.csv'.format(date.today()))