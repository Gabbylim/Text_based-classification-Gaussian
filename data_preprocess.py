import re,string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')

class data_preprocess:
  def __init__(self):
    self.text = ""
    clean_words = []

  def cleaner(self,text ):
    
    text = text.lower()
    text = re.compile("[(+*$!@%&#)]*").sub('',text).strip()
    text = re.sub('\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]',' ',text)
    text=re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text)
    text = re.sub(r'\s+',' ',text)
    text = re.sub(r'\s+',' ',text)
    return text

  def stopword_removal(self,text):
    text = self.cleaner(text)
    #stop word removal
    words = [word for word in text.split() if word.lower() not in    stopwords.words('english')]

    words = ' '.join(words)
    return words

  def lemmatize(self,text):
      
     lem = WordNetLemmatizer()
     text = self.stopword_removal(text)
     lemm_text = [lem.lemmatize(w) for w in  nltk.word_tokenize(text)]
     return ' '.join(lemm_text)

   
    

