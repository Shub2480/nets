# Importing necessary libraries
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Downloading NLTK resources
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk.corpus import stopwords

nltk.download('movie_reviews')
from nltk.corpus import movie_reviews


movie_data = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids()]
movie_labels = [movie_reviews.categories(fileid)[0] for fileid in movie_reviews.fileids()]


# Preprocessing the text (removing stopwords and tokenization)
stop_words = stopwords.words('english')

def preprocess(text):
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(filtered_tokens)

# Preprocess the dataset
movie_data = [preprocess(text) for text in movie_data]
print("len",len(movie_data))


# Vectorizing the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000,ngram_range=(1,2))
X = vectorizer.fit_transform(movie_data)
y = movie_labels

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Evaluating the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["pos","neg"]))

# Save the model and the vectorizer to disk
joblib.dump(model, 'sentiment_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')

print("Model and vectorizer saved successfully!")

