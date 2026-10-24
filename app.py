import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Page Configuration
st.set_page_config(page_title="Sentiment Analysis App", page_icon="📊", layout="centered")

st.title("📊 Sentiment Analysis Major Project")
st.write("Enter a review below to analyze its sentiment using Machine Learning!")

# Train a baseline model on the fly so it works right out of the box
@st.cache_resource
def load_model():
    # Sample training reviews
    texts = [
        "This movie was fantastic, incredible, and amazing!",
        "I loved the acting and the storyline, highly recommended.",
        "Absolute masterpiece, best film I've seen all year.",
        "Terrible, boring, and a complete waste of time.",
        "I hated this movie, the plot made no sense.",
        "Worst film ever, awful acting and terrible direction."
    ]
    labels = [1, 1, 1, 0, 0, 0]  # 1 = Positive, 0 = Negative
    
    # Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(texts)
    
    # Train Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, labels)
    
    return vectorizer, model

vectorizer, model = load_model()

# User input text area
user_input = st.text_area("Enter your review here:", "The movie was absolute fantastic and breathtaking!")

# Prediction button
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Transform user input and predict
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        probability = model.predict_proba(input_vector)[0]
        confidence = max(probability) * 100
        
        # Display results
        if prediction == 1:
            st.success(f"**Result:** Positive Sentiment 😊 (Confidence: {confidence:.2f}%)")
        else:
            st.error(f"**Result:** Negative Sentiment 😞 (Confidence: {confidence:.2f}%)")