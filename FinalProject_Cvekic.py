#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Ivana Cvekic
Topic: Text Classification/Sentiment Analysis
"""

import csv
import pandas as pd
import statistics
from collections import Counter
# NLP libraries
import nltk
import spacy
from spacy import displacy
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

# importing files from folder, no capitalized and no "\n"
# OBAMA
with open('2013-Obama.txt', 'r') as f: 
    obama = f.read().lower().rstrip("\n")
# TRUMP
with open('2017-Trump.txt', 'r') as f: 
    trump = f.read().lower().rstrip("\n")


# find word frequencies in the list
cnt = Counter() # initializing Counter

# create a list from the text 
match_pattern = re.findall('\w+', obama)

# creating function that calculates word frequency
def frequency(text):
    for word in match_pattern:
        if word in text:
            cnt[word] += 1
    print(cnt)

frequency(obama)


# creating a data frame for frequencies (10 most frequent words)
def plot_frequency(text):
    df=pd.DataFrame.from_dict(text, orient = 'index').reset_index()
    df = df.rename(columns={'index':'words', 0:'count'})
    df_count = df.sort_values(by=['count'], ascending=False)
    df_10 = df_count[:10] # only the first 10 words
    # plotting the 10 most frequent words
    df_10.plot(kind='bar',x='words',y='count',color='green') 

plot_frequency(cnt)

# importing stop words: most common/filler words
stop_words = nltk.corpus.stopwords.words("english")


# remove all occurrences of stop words
def removing_stop_words(text):
    for word in text:
        while word in stop_words:
            try:
                text.remove(word)
            except:
                break
    print(len(text))
removing_stop_words(match_pattern)

# new dictionary for visualization
frequencies = {}
def new_frequency(text):

    for word in text:
        if word in frequencies:
            frequencies[word] += 1
        else:
            frequencies[word] = 1
    print(frequencies)
new_frequency(match_pattern)

plot_frequency(frequencies)

# the same procedure for Trump
match_pattern = re.findall('\w+', trump)
frequency(trump)
plot_frequency(cnt)
removing_stop_words(match_pattern)
plot_frequency(frequencies)


# tesing sentence-word radio
# importing sentences
obama_sents = nltk.corpus.inaugural.sents('2013-Obama.txt')
trump_sents = nltk.corpus.inaugural.sents('2017-Trump.txt')

# importing words
obama_words = nltk.corpus.inaugural.words('2013-Obama.txt')
trump_words = nltk.corpus.inaugural.words('2017-Trump.txt')
len(obama_sents)
len(trump_sents)

# ratio words per sentences
def ratio(words, sents):
    ratio = len(words)/len(sents)
    print(f'The ratio words per sentence is {ratio}.')

ratio(obama_words, obama_sents)
ratio(trump_words, trump_sents)


# tokenizing
# importing Spacy English model
nlp = spacy.load('en_core_web_sm')

# how many tokens there are - unique words
obama_tokens = nlp(obama)
trump_tokens = nlp(trump)
len(obama_tokens)
len(trump_tokens)

# choosing one sentence for Obama
obama_one = obama_sents[48]
obama_lemma = ' '.join(obama_one)
obama_lemma = nlp(obama_lemma)

# choosing one sentence for Trump
trump_one = trump_sents[48]
trump_lemma = ' '.join(trump_one)
trump_lemma = nlp(trump_lemma)


# showimg lemmas - root of the word, part of speech and token
def show_lemmas(text):
    for token in text:
        print(f'{token.text:{12}} {token.pos_:{6}} {token.lemma:<{22}} {token.lemma_}')

show_lemmas(obama_lemma)
show_lemmas(trump_lemma)

# showing dependencies - an example of a sentence
# click html: http://127.0.0.1:5000 
displacy.serve(obama_lemma, style='dep')

# click html: http://127.0.0.1:5000 
displacy.serve(trump_lemma, style='dep')

# Named entity recognition
# segmenting a sentence to identify and extract entities
def entities(text):
    for ent in text.ents:
        print(ent.text+' - '+ent.label_+' - '+str(spacy.explain(ent.label_)))

entities(obama_tokens)
entities(trump_tokens)


# Sentiment analysis
# process of taking text and seeing if the language og use is more positive or negative
# via nltk - SentimentIntensityAnalyzer()
sid.polarity_scores(obama)
sid.polarity_scores(trump)


# reading words with a positive meaning
negative = []
with open("words-negative.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        negative.append(row)

print(negative[:10])

# reading words with a negative meaning
positive = []
with open("words-positive.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        positive.append(row)

print(positive[:10])
type(positive)

# definitng sentiment: negative (-1) to positive (1)

def sentiment(text):
    temp = [] # list with sentence sentiment
    text_sent = nltk.sent_tokenize(text)
    for sentence in text_sent: # each sentence in the text starts with n = 0 and p = 0
        n_count = 0
        p_count = 0
        sent_words = nltk.word_tokenize(sentence)
        for word in sent_words: # 
            for item in positive:
                if(word == item[0]):
                    p_count +=1
            for item in negative:
                if(word == item[0]):
                    n_count +=1

        if(p_count > 0 and n_count == 0): #any number of only positives (+) [case 1]
            temp.append(1)
        elif(n_count%2 > 0): # odd number of negatives (-) [case2]
            temp.append(-1)
        elif(n_count%2 == 0 and n_count > 0): #even number of negatives (+) [case3]
            temp.append(1)
        else:
            temp.append(0)
    return temp

# applying the function and averaging all sentences
s_analysis_o = sentiment(obama)
statistics.mean(s_analysis_o)

s_analysis_t = sentiment(trump)
statistics.mean(s_analysis_t)










