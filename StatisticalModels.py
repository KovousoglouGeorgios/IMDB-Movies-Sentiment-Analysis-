import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
import re
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from textblob import TextBlob
import spacy
import os
import wget
import urllib.error

class DataReview:
    """Class for describing the dataset"""

    def __init__(self, data):
        self.data = data

    def describe_dataset(self):
        return self.data.describe()

    def balance_sentiments(self):
        return self.data.iloc[:, -1].value_counts()


class DataCleaner:
    """Class for cleaning and preprocessing text data."""

    @staticmethod
    def strip_html(text):
        if text is not None:
            soup = BeautifulSoup(text, "html.parser")
            return soup.get_text()
        else:
            return ""

    @staticmethod
    def remove_special_characters(text, remove_digits=True):
        if text is not None:
            return re.sub(r'[^a-zA-Z0-9\s]', '', text)
        else:
            return ""

    @staticmethod
    def convert_to_lowercase(text):
        return text.lower() if text is not None else ""


class TextProcessor:
    """Class for text processing tasks such as tokenization, stemming, and stopwords removal."""

    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'\b\w+\b')
        self.snowball_stemmer = SnowballStemmer("english")
        self.stopword_list = set(stopwords.words('english'))

    def snowball_stemmer_function(self, text):
        if text is not None:
            tokens = self.tokenizer.tokenize(text)
            stemmed_text = ' '.join([self.snowball_stemmer.stem(word) for word in tokens])
            return stemmed_text
        else:
            return ""

    @staticmethod
    def spacy_lemmatizer(text):
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(text)
        lemmatized_text = ' '.join([token.lemma_ for token in doc])
        return lemmatized_text

class SentimentAnalyzer:
    def __init__(self):
        # Check if models are downloaded, if not, download them
        models_exist = all(os.path.exists(model_file) for model_file in ['logistic_regression_model.pkl', 'multinomial_naive_bayes_model.pkl', 'svm_model.pkl', 'tfidf_vectorizer.pkl'])
        if not models_exist:
            try:
                self.download_models()
            except urllib.error.HTTPError as e:
                st.error(f"Error downloading models: {e}")
                st.stop()

        # Instantiate data preprocessing classes
        self.data_cleaner = DataCleaner()
        self.text_processor = TextProcessor()
        self.lb = LabelBinarizer()

    @staticmethod
    def download_models():
        base_url = 'https://raw.githubusercontent.com/KovousoglouGeorgios/IMDB-Movies-Sentiment-Analysis-/master/'
        files = ['logistic_regression_model.pkl', 'multinomial_naive_bayes_model.pkl', 'svm_model.pkl', 'tfidf_vectorizer.pkl']
        for file in files:
            wget.download(base_url + file, file)

    def preprocess_comment(self, comment):
        # Apply preprocessing steps
        cleaned_comment = self.data_cleaner.strip_html(comment)
        cleaned_comment = self.data_cleaner.remove_special_characters(cleaned_comment)
        cleaned_comment = self.text_processor.snowball_stemmer_function(cleaned_comment)
        cleaned_comment = self.text_processor.spacy_lemmatizer(cleaned_comment)
        return cleaned_comment

    def predict_sentiment(self, comment):
        # Preprocess the comment
        cleaned_comment = self.preprocess_comment(comment)
        comment_tfidf = self.tv.transform([cleaned_comment])

        # Predict sentiment using each model
        lr_prediction = self.loaded_lr_model.predict(comment_tfidf)
        mnb_prediction = self.loaded_mnb_model.predict(comment_tfidf)
        svm_prediction = self.loaded_svm_model.predict(comment_tfidf)

        return lr_prediction[0], mnb_prediction[0], svm_prediction[0]

# Initialize SentimentAnalyzer
analyzer = SentimentAnalyzer()

# Define function to get sentiment prediction and corresponding color
def cget_sentiment_label(prediction):
    if prediction == 0:
        return "Negative", "#FF5733"  # Change the color code to red
    elif prediction == 1:
        return "Positive", "#2E8B57"    # Change the color code for positive sentiment to green
    else:
        return "Unknown", "#6c757d"     # Add a default case for unknown predictions

# Function to create the Streamlit app
def main():
    # Set up the app title and description
    st.title("Movie Reviews Sentiment Analysis App")
    st.write("This app predicts the sentiment of a given movie review using three statistical algorithms.")

    # Get user input
    user_input = st.text_input("Enter your comment:")

    # Predict sentiment when user clicks the button
    if st.button("Predict Sentiment"):
        if user_input:
            # Get predictions
            lr_pred, mnb_pred, svm_pred = analyzer.predict_sentiment(user_input)

            # Get sentiment label and color for each prediction
            lr_label, lr_color = get_sentiment_label(lr_pred)
            mnb_label, mnb_color = get_sentiment_label(mnb_pred)
            svm_label, svm_color = get_sentiment_label(svm_pred)

            # Display predictions with appropriate colors
            st.write(f"Logistic Regression Prediction: <span style='color:{lr_color}'>{lr_label}</span>", unsafe_allow_html=True)
            st.write(f"Multinomial Naive Bayes Prediction: <span style='color:{mnb_color}'>{mnb_label}</span>", unsafe_allow_html=True)
            st.write(f"SVM Prediction: <span style='color:{svm_color}'>{svm_label}</span>", unsafe_allow_html=True)
        else:
            st.warning("Please enter a comment first!")

if __name__ == "__main__":
    main()
