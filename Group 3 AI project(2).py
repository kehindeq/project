#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[2]:


df_True = pd.read_csv('Desktop/NLP/True.csv',encoding='latin1')


# In[3]:


df_True


# In[4]:


df_True.shape


# In[5]:


df_True.drop_duplicates(inplace = True)


# In[6]:


df_True.shape


# In[7]:


df_True.info()


# In[8]:


df_Fake = pd.read_csv('Desktop/NLP/Fake.csv',encoding='latin1')


# In[9]:


df_Fake


# In[10]:


df_Fake.shape


# In[11]:


df_Fake.drop_duplicates(inplace = True)


# In[12]:


df_Fake.shape


# In[13]:


df_Fake.info()


# In[14]:


df_True['label'] = 1
df_True


# In[15]:


df_Fake['label'] = 0
df_Fake


# In[16]:


df = pd.concat([df_True, df_Fake])


# In[17]:


df


# In[18]:


df.info()


# In[19]:


df['class']=df['label'].map({0:'Fake', 1:True})
df.head(5)


# In[20]:


label_value_counts = df['class'].value_counts()

plt.figure(figsize=(8, 6))
label_value_counts.plot(kind='bar',color=['red','green'])
plt.title('Value Counts of Class Column')
plt.xlabel('Class Values')
plt.ylabel('Counts')
plt.show()


# In[21]:


import nltk
nltk.download('punkt')


# In[22]:


df['no_of_characters'] = df['text'].apply(len)


# In[23]:


df['no_of_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[24]:


df['no_of_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[25]:


df.head()


# In[26]:


df.describe()


# In[27]:


plt.figure(figsize=(8, 6))
sns.boxplot(x='class', y='no_of_characters', data=df)
plt.title('Box Plot of Number of Characters by class')
plt.xlabel('Class')
plt.ylabel('Number of Characters')
plt.show()


# In[28]:


plt.figure(figsize=(8, 6))
sns.boxplot(x='class', y='no_of_words', data=df)
plt.title('Box Plot of Number of Words by Class')
plt.xlabel('Class')
plt.ylabel('Number of Words')
plt.show()


# In[29]:


plt.figure(figsize=(8, 6))
sns.boxplot(x='class', y='no_of_sentences', data=df)
plt.title('Box Plot of Number of Sentences by Class')
plt.xlabel('Class')
plt.ylabel('Number of Sentences')
plt.show()


# In[30]:


plt.figure(figsize=(8, 6))
sns.scatterplot(x='no_of_characters', y='no_of_words', data=df)
plt.title('Scatter Plot of Number of Characters vs Number of Words')
plt.xlabel('Number of Characters')
plt.ylabel('Number of Words')
plt.show()


# In[31]:


df.columns


# In[32]:


df_corr = df[['no_of_characters', 'no_of_words', 'no_of_sentences']].corr()
plt.figure(figsize=(6, 4))
sns.heatmap(df_corr, annot=True, cmap='coolwarm')
plt.title('RFM Correlation Heatmap')
plt.show()


# In[33]:


# Preprocessing using NLTK
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Tokenization, stopword removal, and stemming
def preprocess_text(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    preprocessed_text = ' '.join(stemmed_tokens)
    return preprocessed_text


# In[34]:


df['processed_text'] = df['text'].apply(preprocess_text)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['label'], test_size=0.2, random_state=42)


# In[35]:


# Feature extraction (TF-IDF)
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# In[36]:


# Model training (Logistic Regression)
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)


# In[37]:


# Prediction
y_pred = model.predict(X_test_tfidf)


# In[38]:


# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[47]:


def output_label(n):
    if n ==0:
        return "Fake"
    elif n ==1:
        return "True"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.Dataframe(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_X_test_tfidf = tfidf_vectorizer.transform(new_x_test)
    y_pred = LR.predict(new_X_test_tfidf)
    
    return print("\n\nLR Prediction:".format(output_label(y_pred[0]))


# In[ ]:




