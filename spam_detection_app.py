import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from joblib import load
from sklearn.metrics import accuracy_score
import numpy as np
import gensim

# Load your pre-trained model
model = load("C:/Users/madda/Desktop/Python Projects/spam_detec_model")
ml = gensim.models.Word2Vec.load("C:/Users/madda/Desktop/Python Projects/spam_detec_w2v")
# Define a function to preprocess text data
def pred(text,w2v,lg):
    p = gensim.utils.simple_preprocess(text)
    vectors = [w2v.wv[word] for word in p if word in w2v.wv]
    if vectors:
        res = np.mean(vectors,axis = 0)
    else:
       res = np.zeros(w2v.vector_size)
    fin = lg.predict(res.reshape(1,-1))
    return fin

# Create a Streamlit app
st.title("Spam Detection App")

# Input field for user text
with st.form("text_form"):
    text_input = st.text_area("Enter some text:")
    submit_button = st.form_submit_button("Submit")

if submit_button:
    # Preprocess the input text
    # preprocessed_text = preprocess_text(text_input)

    # Use your model to predict spam or not spam
    # prediction = model.predict()
    prediction = pred(text_input,ml,model)

    # Create a layout to display the result
    with st.expander("Result"):
        if prediction[0] == 1:
            st.write("This text is likely SPAM.")
        else:
            st.write("This text is likely NOT SPAM.")

# Display some example texts for demonstration purposes
st.write("Example Texts:")
st.write(["Hello, how are you?", "This is a spam message.", "I love Python!", "You will never win the lottery."])