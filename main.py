import streamlit as st
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
import pickle
import string

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("Email/SMS spam classifier")
sms=st.text_input("Enter The message:")

if st.button("Predict"):

#Preprocess
    def transform_text(text):
        ps=PorterStemmer()
        text=text.lower()
        text=nltk.word_tokenize(text)
        y=[]
        for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(i)
        text=y.copy()
        y.clear()
        for i in text:
            y.append(ps.stem(i))
        return " ".join(y)
    transform_text=transform_text(sms)
    #2.Vectorize
    vector_input=tfidf.transform([transform_text])


    #3. Predict
    result=model.predict(vector_input)[0]
    #4.Display
    if result==1:
        st.header('SPAM')
    else:
        st.header("Not Spam")