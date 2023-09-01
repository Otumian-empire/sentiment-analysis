# data_collection_process_read_articles.py

# Process the data fetched by
# - removing html tags
# - white spaces
# - and stops word

# python data_collection_process_read_articles.py

import pandas as pd

# collecting data for testing the deep learning learning model

import re

from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import pandas as pd


TEST_FILE_NAME_CSV = "./dev_articles.csv"
FILE_ENCODING = "utf-8"
# Constants
FILE_NAME_PROCESSED_DATA = "preprocess_dev_articles.csv"


# Download stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


# Preprocess text data
def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()

        # Parse the html
        soup = BeautifulSoup(text, "html.parser")

        # Remove special characters and numbers
        clean_from_special_chars = re.sub(
            r"[^a-zA-Z]", " ", "".join(soup.get_text().splitlines()))

        # Tokenization
        tokens = clean_from_special_chars.split()

        # Remove stopwords
        words = [word for word in tokens if word not in stop_words]

        # Join the words back to form a processed sentence
        processed_text = " ".join(words)

        return processed_text

    # Return an empty string for NaN values or non string values
    return ""


# Load article data set
df = pd.read_csv(TEST_FILE_NAME_CSV, names=["text"], encoding="utf-8")

# Shape of dataframe
print("Shape of training dataframe: ", df.shape)
df.info()

# Loop through the DataSet to process it
for index, row in enumerate(df["text"]):
    print(f"✅️ Cleaning {index + 1}")
    df.at[index, "processed_text"] = preprocess_text(row)


# Write only the processed text into file
df = pd.DataFrame(df["processed_text"])
df.to_csv(FILE_NAME_PROCESSED_DATA,  header=False,
          index=False, encoding=FILE_ENCODING)
print(f"Preprocessed text is written into {FILE_NAME_PROCESSED_DATA}")
