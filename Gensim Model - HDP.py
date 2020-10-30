# import all the necessary libraries for LDA Model
import nltk
from nltk.corpus import stopwords
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import HdpModel

# spacy for lemmatization
import spacy

# Plotting tools
import matplotlib.pyplot as plt
from tqdm import tqdm

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# read keyline extracted
data = open('keywords_prison.txt', 'r', encoding = 'utf-8').readlines()
# Get the list of stop words
stop_words = stopwords.words('english')
stop_words.extend(["could","though","would","also","us"])

# define function to tokenize the keyline
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))

print("Tokenized data: ",data_words[:2])

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print("Trigram Sample: ",trigram_mod[bigram_mod[data_words[0]]])

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en_core_web_sm
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv 'NOUN', 'ADJ', 'VERB', 'ADV'
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print("Lemmatized Data: ",data_lemmatized[:1])

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print("Corpus: ",corpus[:1])

# Human readable format of corpus (term-frequency)
print("Corpus with original text: ",[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])

# open file to write output
output = open("hdp_output.txt",'w')
# Loop hdp model with different number of topics (test from 1 to 10)
i=1
while i<11:
    output.write("Number of Topic: ")
    output.write(str(i)+'\n')
    # Build HDP model
    hdp_model = HdpModel(corpus, id2word)
    # Print the Keyword in the 10 topics
    topic_info = hdp_model.print_topics(num_topics=i, num_words=10)
    pprint(topic_info)
    HDP_Topics = topic_info
    for topic in HDP_Topics:
        output.write(' '.join(str(item) for item in topic) + '\n')

    doc_hdp = hdp_model[corpus]    
    # Compute Coherence Score
    coherence_model_hdp = CoherenceModel(model=hdp_model, corpus=corpus, dictionary=id2word, coherence='u_mass')
    coherence_hdp = coherence_model_hdp.get_coherence()
    print('Coherence Score: ', coherence_hdp)
    output.write('Coherence Score: ')
    output.write(str(coherence_hdp)+'\n')
    i+=1
