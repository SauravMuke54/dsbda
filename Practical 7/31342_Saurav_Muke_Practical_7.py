#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np   
import pandas as pd    
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import spacy
nlp = spacy.load('en_core_web_sm')
import document
from docx import Document
import warnings
warnings.filterwarnings('ignore')
import docx2txt


# In[7]:


path = 'testdoc.docx'
text = docx2txt.process(path)


# In[8]:


print(text)


# In[9]:


path = 'testdoc2.docx'
text2 = docx2txt.process(path)


# In[10]:


print(text2)


# In[11]:


doc = nlp(text)
# Extract tokens for the given doc
print ([token.text for token in doc])
print(len(doc))
doc2 = nlp(text2)
print(len(text2))
corpus = [doc,doc2]


# In[12]:


sentences = list(doc.sents)
print(len(sentences))


# In[13]:


for token in doc:
     print ("\n",token, token.idx, token.text_with_ws,
            token.is_alpha, token.is_punct, token.is_space,
            token.shape_, token.is_stop)


# In[14]:


spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

len(spacy_stopwords)


# In[15]:


words = [word.lemma_ for word in doc]
print(words)


# In[16]:


vocabulary = []
vocabulary =  " ".join([word.lemma_ for word in doc if word not in spacy_stopwords])
print(vocabulary)


# In[17]:


for token in doc:
    print(token, token.pos_)


# In[18]:


verbs = [token.text for token in doc if token.pos_ == "VERB"]
nouns = [token.text for token in doc if token.pos_ == "NOUN"]
print('Verbs ',len(verbs),'Nouns ',len(nouns))
print('Verbs ',verbs)


# In[19]:


for token in doc:
    print(token, token.lemma_)


# In[20]:


import nltk
from nltk.stem import WordNetLemmatizer,PorterStemmer


# In[21]:


words = []
 
words = " ".join(token.text for token in doc)
words


# In[22]:


ps = PorterStemmer()
stemmed_words = []
for token in doc:
    stemmed_words.append(ps.stem(token.text))


# In[23]:


stemmed_words


# In[24]:


import nltk
nltk.download('wordnet')


# In[25]:


import nltk
nltk.download('omw-1.4')


# In[26]:


wl = WordNetLemmatizer()
wl_stemmed_words = []
for token in doc:
    wl_stemmed_words.append(wl.lemmatize(token.text))
wl_stemmed_words   


# In[27]:


corpus = [text,text2]
def termfreq(corpus):
    dic={}
    
    for doc in corpus: 
        #words = " ".join([token.text for token in doc])
        for word in doc.split():
            if word in dic:
                dic[word]+=1
            else:
                dic[word]=1
    
    for word,freq in dic.items():
        print(word,freq)        
        dic[word]=freq/len(doc)        
    print('Document size in number of words',len(doc))    
    return dic
termfreq(corpus)


# In[28]:


import re
import nltk
from nltk.corpus import stopwords
doc_text = " "


# In[29]:


import nltk
nltk.download('stopwords')


# In[30]:


def preprocess_docs(text):
    text = str(text).lower()
    #print(text)
    text = re.sub('[^a-zA-z0-9\s]','',str(text))            
    text = text.split()  
    #print(text)
    text = [wl.lemmatize(word) for word in text if not word in stopwords.words('english')]
    new_text = ' '.join(text)
    #print("\ndoc : ", new_text)
        #doc_text = doc_text + new_text
            #doc_text = doc_text + new_text
        #print(new_text)
    return new_text
corpus = [text,text2]
#print(text)
text1 = preprocess_docs(text)
text2 = preprocess_docs(text2)
print("\n doc1",text1,"\nDoc 2",text2)


# In[31]:


first= text1
second= text2
#split so each word have their own string
first = first.split(" ")
second= second.split(" ")
#print(first,second)
total= set(first).union(set(second))
#print(total)
wordDictA = dict.fromkeys(total, 0) 
wordDictB = dict.fromkeys(total, 0)
for word in first:
    wordDictA[word]+=1
    
for word in second:
    wordDictB[word]+=1
#put them in a dataframe and then view the result:
pd.DataFrame([wordDictA, wordDictB])


# In[32]:


def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count/float(bowCount)
    return tfDict
#running our sentences through the tf function:
tfFirst = computeTF(wordDictA, first)
tfSecond = computeTF(wordDictB, second)
#Converting to dataframe for visualization
tf_df= pd.DataFrame([tfFirst, tfSecond])
tf_df.head()


# In[33]:


import math


# In[34]:


def computeIDF(docList):
    idfDict = {}
    N = len(docList)
    
    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1
    
    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / float(val))
        
    return idfDict
#inputing our sentences in the log file
idfs = computeIDF([wordDictA, wordDictB])


# In[35]:


idfs


# In[36]:


def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return tfidf
#running our two sentences through the IDF:
idfFirst = computeTFIDF(tfFirst, idfs)
idfSecond = computeTFIDF(tfSecond, idfs)
#putting it in a dataframe
idf= pd.DataFrame([idfFirst, idfSecond])


# In[37]:


idf.transpose()


# In[ ]:





# In[ ]:




