import streamlit as st
from transformers import BertTokenizer
import tensorflow as tf

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Streamlit app
st.title("Sentiment Analysis with BERT")

# Load the saved model
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = r"C:\Users\User\PycharmProjects\SentimentAnalysis\bert_sentiment_model"
    loaded_model = tf.saved_model.load(model_path)
    return loaded_model

loaded_model = load_model()

# Input text box for user input
user_input = st.text_area("Enter your movie review:", "Type here...")

# Clear the default text when the user clicks on the text area
if user_input == "Type here...":
    user_input = ""

# Button to perform sentiment analysis
if st.button("Analyze Sentiment"):
    if user_input:
        # Tokenize input text
        tf_batch = tokenizer([user_input], max_length=128, padding=True, truncation=True, return_tensors='tf')

        # Make predictions with the fine-tuned model
        tf_outputs = loaded_model(tf_batch)
        tf_predictions = tf.nn.softmax(tf_outputs['logits'], axis=-1)

        # Define class labels
        labels = ['Negative', 'Positive']

        # Extract predicted label
        label = tf.argmax(tf_predictions, axis=1).numpy()[0]

        # Display the result with colored text
        if label == 0:
            st.write(f"<span style='color:red'>Prediction: {labels[label]}</span>", unsafe_allow_html=True)
        else:
            st.write(f"<span style='color:green'>Prediction: {labels[label]}</span>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a movie review.")

# Footer
st.markdown("Built with TensorFlow, Keras, Transformers, and Streamlit.")
