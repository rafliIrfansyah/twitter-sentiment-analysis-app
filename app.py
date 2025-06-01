import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from joblib import load as loadVectorizer


# Function to clean tweets
def clean_tweet(tweet):
    # Remove URLs
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    # Remove user mentions
    tweet = re.sub(r'@\w+', '', tweet)
    # Remove hashtags
    tweet = re.sub(r'#\w+', '', tweet)
    # Remove numbers
    tweet = re.sub(r'\d+', '', tweet)
    # Remove special characters
    tweet = re.sub(r'[^\w\s]', '', tweet)
    # Convert to lowercase
    tweet = tweet.lower().strip()
    return tweet

# Load the dataset
@st.cache_data
def load_data():
    columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
    df = pd.read_csv('dataset/training.1600000.processed.noemoticon.csv',
                    encoding='latin-1',
                    names=columns)
    df['target'] = df['target'].map({0: 0, 4: 1})  # Convert 0,4 to 0,1
    return df

# Load the pre-trained model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    with open('model/svm_model.pkl', 'rb') as file:
        model = pickle.load(file)
    # Since the vectorizer wasn't saved, we need to recreate it
    # Note: In a production app, you should save and load the vectorizer too
    df = load_data()
    df['cleaned_text'] = df['text'].apply(clean_tweet)
    vectorizer = loadVectorizer('util/tfidf_vectorizer.joblib')
    return model, vectorizer

# Streamlit app
st.title("Twitter Sentiment Analysis Application")

# Create tabs
tabs = ["Data Description", "Data Pre-processing", "Prediction", "Prediction Results"]
tab1, tab2, tab3, tab4 = st.tabs(tabs)

# Tab 1: Data Description
with tab1:
    st.header("Data Description")
    df = load_data()
    st.write("Dataset Overview:")
    st.dataframe(df.head())
    st.write(f"Dataset Shape: {df.shape}")
    st.write("Target Distribution:")
    st.bar_chart(df['target'].value_counts())

# Tab 2: Data Pre-processing
with tab2:
    st.header("Data Pre-processing")
    st.write("Steps in Pre-processing:")
    st.markdown("""
    1. Remove URLs
    2. Remove user mentions (@username)
    3. Remove hashtags
    4. Remove numbers
    5. Remove special characters
    6. Convert lowercase
    """)
    st.write("Sample Pre-processing:")
    sample_text = df['text'].iloc[0]
    cleaned_sample = clean_tweet(sample_text)
    st.write("Original Text:", sample_text)
    st.write("Cleaned Text:", cleaned_sample)

# Tab 3: Prediction
with tab3:
    st.header("Make a Prediction")
    model, vectorizer = load_model_and_vectorizer()
    
    # Input for new text
    new_text = st.text_area("Enter your text here:", "I love this beautiful day!")
    
    if st.button("Predict"):
        # Clean the input text
        cleaned_text = clean_tweet(new_text)
        # Transform the text using the vectorizer
        text_vector = vectorizer.transform([cleaned_text])
        # Make prediction
        prediction = model.predict(text_vector)
        prediction_proba = model._predict_proba_lr(text_vector)
        # Store results in session state
        st.session_state['prediction'] = prediction
        st.session_state['prediction_proba'] = prediction_proba
        st.session_state['cleaned_text'] = cleaned_text
        st.session_state['original_text'] = new_text
        st.success("Prediction completed! Check the 'Prediction Results' tab.")

# Tab 4: Prediction Results
with tab4:
    st.header("Prediction Results")
    if 'prediction' in st.session_state:
        st.write("**Original Input Text:**")
        st.write(st.session_state['original_text'])
        st.write("**Pre-processed Text:**")
        st.write(st.session_state['cleaned_text'])
        st.write("**Prediction:**")
        prediction = st.session_state['prediction']
        prediction_proba = st.session_state['prediction_proba']
        label = "Positive" if prediction == 1 else "Negative"
        st.write(f"Negative Probability: **{prediction_proba[0][0]}**")
        st.write(f"Positive Probability: **{prediction_proba[0][1]}**")
        st.write(f"Sentiment: **{label}**")

    else:
        st.write("No prediction yet. Please make a prediction in the 'Prediction' tab.")

if __name__ == "__main__":
    print('App is Running Correctly')