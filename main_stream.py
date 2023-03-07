from data_preprocess import data_preprocess
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
import pandas as pedi

df = pedi.read_csv('./Corona_NLP_test.csv')
process = data_preprocess()


df['OriginalTweet'] = df['OriginalTweet'].apply(lambda x : process.lemmatize(x))

#for i in df.iterrows():
 # lambda x : lemmatize(df['OriginalTweet'][i])
#print(df['OriginalTweet'])
X_train,X_test,y_train,y_test = train_test_split(df['OriginalTweet'],df['Sentiment'],test_size =  0.66,random_state=6)
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train,y_train)
predicted = model.predict(X_test)
print("Accuracy score is {}".format(accuracy_score(y_test,predicted)))
