# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 22:40:15 2022

@author: Vishal
"""
import pandas as pd
from nltk.corpus import stopwords 
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
stop_words = set(stopwords.words('english')) 

pd.set_option('display.max_rows', 371)
pd.set_option('display.max_columns', 10)

# Loading Data-set in Panda dataFramew
data_fram = pd.read_csv ('C:\D Drive\Centennial\sem 1\AI\Project\Scripts\Youtube05-Shakira.csv')
data_fram.shape
data_fram= data_fram.drop(['COMMENT_ID', 'AUTHOR','DATE'], axis=1)
data_fram.shape


shuffle_data = data_fram.sample(frac = 1)
train_percent = 75
train_data = shuffle_data[:int(len(shuffle_data)*(train_percent/100))]
test_data = shuffle_data[int(len(shuffle_data)*(train_percent/100)):]


count_vectorizer = CountVectorizer()
train_Vectorized_data = count_vectorizer.fit_transform(train_data["CONTENT"])
test_Vectorized_data = count_vectorizer.transform(test_data["CONTENT"])
train_Vectorized_data.shape
print("\nDimensions of training data:", train_Vectorized_data.shape)

imp_features = count_vectorizer.get_feature_names()
print(imp_features)
print(type(train_Vectorized_data))

tf_idf = TfidfTransformer()
train_final = tf_idf.fit_transform(train_Vectorized_data)
test_final = tf_idf.fit_transform(test_Vectorized_data)
train_final.shape
print(type(train_final))


train_labels = train_data["CLASS"]
test_labels = test_data["CLASS"]

print(train_final.shape)
print(test_final.shape)

nb_classifier = MultinomialNB().fit(train_final, train_labels)

folds = 5
accuracy = cross_val_score(nb_classifier, train_final.toarray(), train_labels, scoring='accuracy', cv=folds)
print("Avg Accuracy: " + str(round(100*accuracy.mean(), 2)) + "%")


data_predictions = nb_classifier.predict(test_final)
print("Accuracy : ", accuracy_score(test_labels, data_predictions))
print(" Confusion Matrix ")
print(confusion_matrix(test_labels, data_predictions))
print(" Report")
print(classification_report(test_labels, data_predictions))


input_comments = ['Awsome!', 'nice', 'Amazing Dance','Hey Can my friend can have car???','my friend is in love with you','subscirbe to my channel']
comments = input_comments
input_comments = count_vectorizer.transform(input_comments)   
type(input_comments)
print(input_comments)
input_comments = tf_idf.transform(input_comments)
print(input_comments)

result = nb_classifier.predict(input_comments)
print(result)

for comment, category in zip(comments, result):
    if category==0:
        print(comment+' is not a spam')
    else: 
        print(comment+" is a spam")








