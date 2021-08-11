import sys
import re
import nltk
import numpy as np
import pickle
from flask import Flask,render_template,url_for,request

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()

o = open("stopwords.pkl", "rb")
stp = pickle.load(o)

app = Flask(__name__)

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
  corpus1 = []
  if request.method == 'POST':
    message = request.form['message']
    data = message
    review = re.sub('[^a-zA-Z]', ' ', data)
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stp]
    review = ' '.join(review)
    corpus1.append(review)

    m = open("tfidf.pkl", "rb")
    tfidf = pickle.load(m)
    X1 = tfidf.transform(corpus1)
    n = open("model.pkl", "rb")
    clf = pickle.load(n)
    output = clf.predict(X1)
  return render_template('result.html',prediction = output[0])


if __name__ == '__main__':
  app.run(debug=True)