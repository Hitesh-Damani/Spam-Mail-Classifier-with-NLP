# -*review=- coding: utf-8 -*-
"""
Created on Tue Jan 21 19:43:36 2020

@author: Master
"""

import pandas as pd


messages=pd.read_csv('E:/Natural Language Processing/SMSSpamCollection', sep='\t',names=['labels','messages'])

# Data Cleaning and Preprocessing
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
ps=PorterStemmer()
wordnet=WordNetLemmatizer()
corpus=[]

for i in range(0,len(messages)):
    review=re.sub('[^a-zA-Z]',' ' , messages['messages'][i])
    review=review.lower()
    review=review.split()
    review=[wordnet.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)
    
# Creating Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2500)
X=cv.fit_transform(corpus).toarray()
 
# Creating dummies of label column
y=pd.get_dummies(messages['labels'])
y=y.iloc[:,1].values

# Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

# Training model using Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect_model=MultinomialNB().fit(X_train,y_train)

# Predicting the output
y_predict=spam_detect_model.predict(X_test)

# Compairing the model
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predict)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_predict)
