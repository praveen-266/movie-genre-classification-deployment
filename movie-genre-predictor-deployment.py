import numpy as np
import pandas as pd
import pickle

# Loading the dataset
train=pd.read_csv("C:/users/prave/datasets/csv files/NLP dataset/movie genre classification/kaggle_movie_train.txt")

# Mapping the "genre" column into values
genre_mapper={'other':0,'action':1,'adventure':2,'comedy':3,'drama':4,'horror':5,'romance':6,'sci-fi':7,'thriller':8}
train['genre']=train['genre'].map(genre_mapper)

# Removing "id" columns
train.drop(['id'],axis=1,inplace=True)

# importing essential libraries for performing Natural Language Preprocessing
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Cleaning the text
corpus=[]
ps=PorterStemmer()

for i in range(0,train.shape[0]):
    
    # cleaning special character fro the script
    script=re.sub(pattern='[^a-zA-Z]',repl=' ',string=train['text'][i])
    
    # Converting the entire text into lower case
    script=script.lower()
    
    # tokenizing the script by words
    words=script.split()
    
    # Removing the stop words
    script_words=[word for word in words if word not in set(stopwords.words('english'))]
    
    # stemming the stop words
    words=[ps.stem(word) for word in script_words]
    
    # joining the stemmed words
    script=' '.join(words)
    
    #creating a corpus
    corpus.append(script)
    

# Creating the Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=10000)
X=cv.fit_transform(corpus).toarray()
y=train['genre'].values

# creating a pickle file for the Countvectorizer
pickle.dump(cv,open('cv-transformer.pkl','wb'))

# Model Building


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import MultinomialNB
nb_classifier = MultinomialNB(alpha=0.1)
nb_classifier.fit(X_train, y_train)

# creating a pickle file for the Mutinomial NAive Bayes model
filename='movie-genre-mnb-model.pkl'
pickle.dump(nb_classifier,open(filename,'wb'))