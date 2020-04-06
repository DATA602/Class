#!/usr/bin/env python
# coding: utf-8

# In[30]:


import urllib.request
import os
import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


# In[31]:


df = pd.read_csv('e:/Machine Learning/movie_data.csv', encoding='utf-8')
df.head(3)


# In[32]:


import nltk
nltk.download('stopwords')
# init Objects
tokenizer=RegexpTokenizer(r'\w+')
en_stopwords=set(stopwords.words('english'))
ps=PorterStemmer()
def getStemmedReview(review):
    review=review.lower()
    review=review.replace("<br /><br />"," ")
    #Tokenize
    tokens=tokenizer.tokenize(review)
    new_tokens=[token for token in tokens if token not in  en_stopwords]
    stemmed_tokens=[ps.stem(token) for token in new_tokens]
    clean_review=' '.join(stemmed_tokens)
    return clean_review


# In[33]:


df['review'].apply(getStemmedReview)
X_train = df.loc[:35000, 'review'].values
y_train = df.loc[:35000, 'sentiment'].values
X_test = df.loc[35000:, 'review'].values
y_test = df.loc[35000:, 'sentiment'].values


# In[34]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, encoding='utf-8',
 decode_error='ignore')
vectorizer.fit(X_train)
X_train=vectorizer.transform(X_train)
X_test=vectorizer.transform(X_test)


# In[35]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver='liblinear')
model.fit(X_train,y_train)
print("Score on training data is: "+str(model.score(X_train,y_train)))
print("Score on testing data is: "+str(model.score(X_test,y_test)))


# In[36]:


# Here 0 denotes a negative sentiment
model.predict(X_test[0])
# 78% probability that the given text is negative
model.predict_proba(X_test[0])

