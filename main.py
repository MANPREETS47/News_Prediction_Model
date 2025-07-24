import pickle
import streamlit as st
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk 

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

with open('fake_real_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('tf_idf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

st.title('Fake or Real News Classifier')
st.write('This app classifies news articles as either fake or real.')

text = st.text_area('Enter the news article text here:')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+','',text)
    text = re.sub(r'[^\w\s]','',text)
    text = re.sub(r'\s+',' ',text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

article = clean_text(text)
# if not article:
#     st.warning('Please enter some text to classify.')
#     st.stop()
if article:
    article = vectorizer.transform([article])



if st.button('Classify'):
    prediction = model.predict(article)
    if prediction[0] == 1:
        st.write('This article is **REAL**.')
    else:
        st.write('This article is **FAKE**.')