# collecting data for testing the deep learning learning model

import os
import re
import time

from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import pandas as pd
import requests
from dotenv import load_dotenv


# load data from .env file
load_dotenv()


# request constants
DEV_TO_URL = "https://dev.to/api/articles"
API_KEY = os.environ.get("API_KEY")
REQUEST_WAIT_TIME = 0.2
TEST_FILE_NAME_CSV = "test_dev_articles.csv"
FILE_ENCODING = "utf-8"

# Download stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Manually curated test articles
path_to_test_articles = [
    "/lissy93/cli-tools-you-cant-live-without-57f6",
    "/jeroendedauw/advice-for-junior-developers-30am",
    "/novu/master-notifications-with-chatgpt-react-and-nodejs-1ea9",
    "/shnai0/how-i-build-my-first-open-source-project-with-chatgpt-and-nextjs-10k-users-in-24-hours-2m7n",
    "/this-is-learning/backend-development-is-more-than-writing-endpoints-for-frontend-gl1",
    "/novu/building-a-notion-like-system-with-socketio-and-react-1hjg",
    "/danielhe4rt/database-101-how-does-migration-between-databases-work-in-the-real-world-24dn",
    "/bobbyiliev/how-to-join-mysql-and-postgres-in-a-live-materialized-view-5864",
    "/mohammadfaisal/21-best-practices-for-a-clean-react-project-jdf",
    "/christinec_dev/lets-learn-godot-4-by-making-an-rpg-part-5-setting-up-the-game-gui-1-186m",
]


# Preprocess text data
def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and numbers
        text = re.sub(r"[^a-zA-Z]", " ", text)

        # Tokenization
        words = text.split()

        # Remove stopwords
        words = [word for word in words if word not in stop_words]

        # Join the words back to form a processed sentence
        processed_text = " ".join(words)

        return processed_text
    else:
        return ""  # Return an empty string for NaN values


test_articles = []

for index, url_path in enumerate(path_to_test_articles):
    article_path = f"{DEV_TO_URL}{url_path}"

    # Get the articles body
    article_response = requests.get(
        article_path, params={
            "api_key": API_KEY
        }
    )

    # On success extract the html body
    if article_response.status_code == 200:
        article_response_data = article_response.json()
        print(str(index + 1), "read::", article_response_data["title"])
        body_html = article_response_data["body_html"]

        # parse the html
        soup = BeautifulSoup(body_html, "html.parser")
        test_articles.append(
            preprocess_text("".join(soup.get_text().splitlines())))

    # wait for a millisecond before making the next request
    time.sleep(REQUEST_WAIT_TIME)


# print(test_articles)
# Write the data response into a csv file using pandas
# Create a pandas DataFrame from the cleaned content
df = pd.DataFrame(test_articles)

# Write the DataFrame to a CSV file
df.to_csv(TEST_FILE_NAME_CSV, index=True, encoding=FILE_ENCODING)

print("Test data collected successfully")
