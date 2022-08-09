from json import load
import pickle
from pyexpat import model
import string
#from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
#from sklearn.feature_extraction.text import CountVectorizer

from flask import Flask, request


app = Flask(__name__)

#cv = CountVectorizer() #inistializing CountVectorizer

#Load Model
def load_model():
    filename = 'Sentiment_Model.pkl'
    model = pickle.load(open(filename, 'rb'))
    return model

#Load CountVectorizer
def load_CV_model():
    filename = 'CountVectorizer.pkl'
    cv_model = pickle.load(open(filename, 'rb'))
    return cv_model

#Check for Fake Reviews
def fake_test(text, rating):
    if 'free sample' in text and rating > 3:
        return 1 #it's a fake review
    else:
        return 0 #review is not fake

#Clean the Text
def clean_text(text):
    
    """
    Takes in a string of text, then preforms the following:
    1. Remove all punctuation
    2. Remove Numbers and Special characters
    3. Remove all stepwords
    4. Return a list of the cleaned text
    """

    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in str(text).split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    #remove special characters
    special = [value for value in text if value.isalnum() or value == ' ']
    special = ''.join(special)
    # remove stop words
    stop = nltk.corpus.stopwords.words('english')
    #stop = nltk.stopwords.words('english')
    text = [x for x in text if x not in stop]
    # join all
    text = " ".join(text)
    return(text)


@app.route('/predict', methods=["POST", "GET"])
def predict():
    model = load_model()
    cv = load_CV_model()

    request_json = request.get_json()
    text = (request_json['Review'])
    rating = int(request_json['Rating'])
    x = fake_test(text, rating)
    if x == 1:
        return {"result": "This review was fake"}

    text = clean_text(text)
    text = cv.transform([text]) #Transform the text using CountVectorizer
    test = model.predict(text)

    if test == 1:
        return {"response": "Positive Sentiment"}
    elif test == 0:
        return {"response": "Negative Sentiment"}


if __name__ == '__main__':
    app.run(debug=True)