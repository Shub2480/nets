# inference.py

import joblib
import nltk
from nltk.corpus import stopwords

# Downloading NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the saved model and vectorizer
model = joblib.load('sentiment_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# Sample inference data (new, unseen text)
inference_data = [
    "This is an amazing movie!",
    "The movie was very bad",
]

# Preprocessing the inference data
stop_words = stopwords.words('english')

def preprocess(text):
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(filtered_tokens)

# Preprocess the inference data
inference_data_processed = [preprocess(text) for text in inference_data]

# Vectorize the inference data using the loaded vectorizer
X_inference = vectorizer.transform(inference_data_processed)

# Use the loaded model to make predictions
predictions = model.predict(X_inference)

# Print the predictions
for text, prediction in zip(inference_data, predictions):
    print(f"Text: {text}")
    print(f"Predicted category: {prediction}")
    print("-" * 50)
