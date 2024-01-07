import streamlit as st
from transformers import BertTokenizer
import tensorflow as tf
from tensorflow import keras

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load the saved model
model = keras.models.load_model(r"C:\Users\User\PycharmProjects\pythonProject-streamlit\bert_sentiment_model")

import os

model_directory = "C:/Users/User/PycharmProjects/pythonProject-streamlit/bert_sentiment_model"

# Print the contents of the directory
print(os.listdir(model_directory))

# Streamlit app
st.title("Sentiment Analysis with BERT")

# Input text box for user input
user_input = st.text_area("Enter your movie review:", "Type here...")

# Button to perform sentiment analysis
if st.button("Analyze Sentiment"):
    if user_input:
        # Tokenize input text
        tf_batch = tokenizer([user_input], max_length=128, padding=True, truncation=True, return_tensors='tf')

        # Make predictions with the fine-tuned model
        tf_outputs = model(tf_batch)
        tf_predictions = tf.nn.softmax(tf_outputs['logits'], axis=-1)

        # Define class labels
        labels = ['Negative', 'Positive']

        # Extract predicted label
        label = tf.argmax(tf_predictions, axis=1).numpy()[0]

        # Display the result
        st.write(f"Prediction: {labels[label]}")
    else:
        st.warning("Please enter a movie review.")

# Footer
st.markdown("Built with TensorFlow, Keras, Transformers, and Streamlit.")
