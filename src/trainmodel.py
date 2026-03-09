import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

# Load cleaned dataset
df = pd.read_csv(r"C:\Users\K HARIKA\OneDrive - Alliance University\Desktop\capstone\Data\cleaned_tweets.csv")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["sentiment"], test_size=0.2, random_state=42
)

# Convert text to numbers
vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict
predictions = model.predict(X_test_vec)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy)

# Save model
pickle.dump(model, open("models/naive_bayes.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("Model saved successfully")