#!/usr/bin/env python
# coding: utf-8

# Loading needed libraries or packages

# In[1]:


import spacy
import re
import csv
import os
import json

# from google.colab import drive


# Mounting Google drive

# In[ ]:



drive.mount('/content/drive')


# Variables initialization

# In[1]:


nlp = spacy.load("en_core_web_sm")
stopwords = nlp.Defaults.stop_words

inverted_idx = {}
documentStore = {}


# Definition of working directory and file names

# In[ ]:


newsDirectory = "/content/drive/MyDrive/BIA-660-Team-Project"
fileNameIndex = 'A_bitcoin_news_Index.csv'
fileNameDocuStore = 'A_bitcoin_news_docu_store.csv'
fileNewsList = []


# In[ ]:


fileNameIndex = os.path.join(newsDirectory, fileNameIndex)
fileNameDocuStore = os.path.join(newsDirectory, fileNameDocuStore)


# Function 1

# In[ ]:



def generateindexifnotexist(filenamein, filenameindex, filenamedocustore):
    global inverted_idx
    global documentStore

    try:
        os.remove(filenameindex)
    except:
        print(f"     File {filenameindex} doesn't exist. Trying to clean next.")
    else:
        pass

    try:
        os.remove(filenamedocustore)
    except:
        print(f"     File {filenameindex} doesn't exist. Now indexing...\n")
    else:
        pass

    print(f"Processing file {os.path.basename(fileNameIn)}")
    with open(filenamein, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",")

        for i, row in enumerate(reader):

            row_tokens = nlp(
                       re.sub(r'<a href.*\">', '',
                                    re.sub(r"</a>", " ",
                                             str(" ".join(row[5:])).lower()
                                           )
                                    )
            )

            document = [token.text for token in row_tokens if ( not token.is_stop and not re.match(r'\W+', token.text) )]
            document = list(set(document))

            # print(document)

            for term in document:
                if term not in inverted_idx:
                    inverted_idx[term] = []
                inverted_idx[term].append(os.path.basename(fileNameIn))
            documentStore[os.path.basename(fileNameIn)] = " ".join(row)
        f.close()

    with open(filenameindex, "w", encoding="utf-8", newline='') as f:
        print(f"     Writing the {filenameindex}")
        json.dump(inverted_idx, f, indent=4)
        f.close()

    with open(filenamedocustore, "w", encoding="utf-8", newline='') as f:
        print(f"     Writing the {filenamedocustore}")
        json.dump(documentStore, f, indent=0)
        f.close()


# Function 2 - Read from file Index and save in memory

# In[ ]:


def readindex(filenameindex):
    with open(filenameindex, "r", encoding="utf-8", newline='') as f:
        print(f"Reading {filenameindex}")
        inverted_idx = json.load(f)
        f.close()
        print("    Index file loaded \n")
        return inverted_idx


# Function 3 - Read from file docuStore and sabe in memory

# In[ ]:


def readdocstore(filenamedocustore):
    with open(filenamedocustore, "r", encoding="utf-8", newline='') as f:
        print(f"Reading {filenamedocustore}")
        documentstore = json.load(f)
        f.close()
        print("    Document Store loaded \n")
        return documentstore


# ####### Main Program

# Create Index File or Docu Store if those doesnt exits in disc.

# In[ ]:


print("Checking if Index and Document Store exists.")
if not os.path.exists(fileNameIndex) or not os.path.exists(fileNameDocuStore):
    print("One or the two files doesn't exist. Starting the Indexing...\n")

    print("Reading news files in directory \n")

    for file in os.listdir(newsDirectory):
        if file.startswith("binance") and file.endswith(".txt"):
            fileNewsList.append(file)

    print(f"Total files being found: {len(fileNewsList)} \n")

    for fileNameIn in fileNewsList:
        fileNameIn = os.path.join(newsDirectory, fileNameIn)
        if os.path.exists(fileNameIn):
            generateindexifnotexist(fileNameIn, fileNameIndex, fileNameDocuStore)
        else:
            print(f"File {fileNameIn} to be indexed doesn't exist. Trying with next file\n")

else:
        print("    Index and Document Store exists.\n")


# Load into memory the Index and Docu Store from file.

# In[ ]:


print("Checking if Index and Document Store are loaded to memory.\n")
if len(documentStore) == 0 or len(inverted_idx) == 0:
    print("Reading last created Index and Document Store files \n")
    inverted_idx = readindex(fileNameIndex)
    documentStore = readdocstore(fileNameDocuStore)
else:
    print("    Already loader.\n")


# Use the Search

# In[ ]:


searchWord = 'x'
while searchWord is not None:
    searchWord = input("\n Write the query word (to quit type 'quit()'):").strip()
    if searchWord == 'quit()':
      print("Exiting the program")
      break
    
    print(f"\n The news with the word {searchWord} are,\n")
    # pdb.set_trace()
    try:
        for i in range(len(inverted_idx[searchWord.lower()])):
            value = str(inverted_idx[searchWord.lower()][i-1])
            print(f"    > {value} : {documentStore[value]}")
    except KeyError as e:
        print (f"     Value {e} doesnt exist in the index")


# ---- TOPIC MODELING -----

# In[ ]:


doc_list  = []
file_name_list = []

path = "/content/drive/My Drive/BIA-660-Team-Project/"

def read_text_file(file_path):
    with open(file_path, 'r') as f:
        doc_list.append(f.read())

os.chdir(path)       
        


# In[ ]:


for file in os.listdir():
    # Check whether file is in text format or not
    if file.endswith(".txt"):
        file_path = f"{path}{file}"
        # call read text file function
        read_text_file(file_path)
        file_name_list.append(file)


# In[ ]:


len(doc_list)


# In[ ]:


file_name_list[0]


# In[ ]:


pip install pyldavis


# In[ ]:


import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
#import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim_models  # don't skip this
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

import nltk; nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from nltk import tokenize
nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# In[ ]:


data_words_list = []

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

for lines in doc_list:
    data_words_list.append(list(sent_to_words(tokenizer.tokenize(lines))))


# In[ ]:


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return 

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[ ]:


import spacy

spacy.load("en_core_web_sm")


# In[ ]:


lda_model_list = []
corpus_list = []
id2word_list = []

k = 0
def prepare_data_for_lda(data_words):
    global k
    k= k+1
    print(k)
    print("Size of data_word:" + str(len(data_words)))
    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out
    
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = [bigram_mod[doc] for doc in data_words_nostops]

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    id2word_list.append(id2word)
    

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    corpus_list.append(corpus)
    
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=2, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
    lda_model_list.append(lda_model)
    
    


# In[ ]:


for data_words in data_words_list:
    prepare_data_for_lda(data_words)


# In[ ]:


len(lda_model_list)


# In[ ]:


print('\nPerplexity: ', lda_model_list[1].log_perplexity(corpus_list[1])) 
# a measure of how good the model is. lower the better.


# In[ ]:


pprint(lda_model_list[2].print_topics())
doc_lda = lda_model_list[2][corpus_list[2]]

print(lda_model_list[3])


# In[ ]:


with open('topics.txt', 'w') as f:

    for i, file_name in enumerate(file_name_list):

        x=lda_model_list[i].show_topics(num_topics=2, num_words=20,formatted=False)
        topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in x]

        #Below Code Prints Topics and Words
        for topic,words in topics_words:
            f.write(file_name + "::" + str(topic)+ "::"+ str(words))
            f.write("\n")
    f.close()


# In[ ]:


import pyLDAvis.gensim_models as gensimvis

# Visualize the topics
pyLDAvis.enable_notebook()
vis = gensimvis.prepare(lda_model_list[3], corpus_list[3], id2word_list[3])
vis


# In[ ]:


from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers = 20) as executor:
      executor.map(prepare_data_for_lda, data_words_list)


# In[ ]:


from multiprocessing import Pool

with Pool(processes=40) as pool:
  pool.map(prepare_data_for_lda, data_words_list)


# In[ ]:


data_words_list[1]

