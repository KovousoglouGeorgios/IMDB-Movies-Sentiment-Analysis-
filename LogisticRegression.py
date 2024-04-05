import streamlit as st
import joblib
import gdown  # Add this import
from bs4 import BeautifulSoup
import re
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import spacy

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
        # Download the saved models from Google Drive
        gdown.download("https://drive.google.com/uc?id=1v-rJHs4nUZqO24Rjhkda47ig5iw_McGU", "logistic_regression_model.pkl", quiet=False)
        gdown.download("https://drive.google.com/uc?id=10OlIZNHKIY2nBlF85vfaurNqWwXMI-TU", "tfidf_vectorizer.pkl", quiet=False)

        # Load the saved models
        self.loaded_lr_model = joblib.load('logistic_regression_model.pkl')

        # Load TF-IDF vectorizer
        self.tv = joblib.load('tfidf_vectorizer.pkl')

        # Instantiate data preprocessing classes
        self.data_cleaner = DataCleaner()
        self.text_processor = TextProcessor()

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

        # Predict sentiment using logistic regression model
        lr_prediction = self.loaded_lr_model.predict(comment_tfidf)

        return lr_prediction[0]

# Initialize SentimentAnalyzer
analyzer = SentimentAnalyzer()

# Define function to get sentiment prediction and corresponding color
def get_sentiment_label(prediction):
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
    st.write("This app predicts the sentiment of a given movie review using logistic regression model.")

    # Get user input
    user_input = st.text_input("Enter your comment:")

    # Predict sentiment when user clicks the button
    if st.button("Predict Sentiment"):
        if user_input:
            # Get predictions
            lr_pred = analyzer.predict_sentiment(user_input)

            # Get sentiment label and color for the prediction
            lr_label, lr_color = get_sentiment_label(lr_pred)

            # Display prediction with appropriate color
            st.write(f"Logistic Regression Prediction: <span style='color:{lr_color}'>{lr_label}</span>", unsafe_allow_html=True)
        else:
            st.warning("Please enter a comment first!")

if __name__ == "__main__":
    main()
