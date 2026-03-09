import pandas as pd
import re

# Load dataset
df = pd.read_csv(r"C:\Users\K HARIKA\OneDrive - Alliance University\Desktop\capstone\Data\training.1600000.processed.noemoticon.csv", encoding='latin-1', header=None)

# Select only sentiment and tweet text
df = df[[0,5]]
df.columns = ["sentiment","text"]

# Convert labels (4 → 1)
df["sentiment"] = df["sentiment"].replace({4:1})

# Clean tweets
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+","",text)
    text = re.sub(r"[^a-zA-Z ]","",text)
    return text

df["text"] = df["text"].apply(clean_text)

# Take 50k tweets
df = df.sample(50000)

# Save cleaned dataset
df.to_csv("data/cleaned_tweets.csv",index=False)

print("Preprocessing Completed")