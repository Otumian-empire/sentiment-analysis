# data_collection_read_articles_by_url_path.py

# Make API requests to read all the articles with the url path written into
# article_paths.txt.
# At the same time strip the content of html characters and clean it using
# `preprocess_text` function
# python data_collection_read_articles_by_url_path.py


# Data collection from dev.to
import aiohttp
import asyncio
from nltk.corpus import stopwords
import nltk
import time
import os
from dotenv import load_dotenv
import pandas as pd
import time


# load data from .env file
load_dotenv()


# Request constants
API_KEY = os.environ.get("API_KEY")

if API_KEY == None or len(API_KEY) < 1:
    print("Please create a `.env` file in the root of the application and add the `API_KEY`")
    exit()


# Download stopwords
nltk.download("stopwords")
STOP_WORDS = set(stopwords.words("english"))

# Base api url
DEV_TO_URL = "https://dev.to/api/articles"

# File constants
# input: read from
FILE_NAME_URL_PATHS = "article_paths.txt"
# output: write to
FILE_NAME_CSV = "dev_articles.csv"
# csv data encoding
FILE_ENCODING = "utf-8"

# Request wait time
REQUEST_WAIT_TIME = 0.2

# DataFrame columns
DATA_FRAME_COLUMNS = ["text", "processed_text", "sentiment"]

# FILE MODES
READ_MODE = "r"
WRITE_MODE = "+w"
APPEND_MODE = "+a"


# STEP and END determines the total number of data to read by doing
# END * STEP
END = 1000
STEP = 100
START = 0


# Get the html body from the success body
async def get_html_body_from_success_response(session, url):
    async with session.get(url) as response:
        if response.status == 200:
            article_data = await response.json()
            print("✅️", article_data.get("title", ""))
            body_html = article_data.get("body_html", "")

            return " ".join(body_html.splitlines())

        return ""


# Fetch a batch of articles
async def fetch_articles(url_paths):
    async with aiohttp.ClientSession() as session:
        tasks = []

        for article_path in url_paths:
            article_url = f"{DEV_TO_URL}{article_path.strip()}"

            task = asyncio.create_task(
                get_html_body_from_success_response(session, article_url))

            tasks.append(task)

        responses = await asyncio.gather(*tasks)
        return responses


try:
    start_time = time.time()  # Record the start time
    print("Starting")

    # Load url paths
    with open(FILE_NAME_URL_PATHS, "r") as url_path_file:
        url_paths = url_path_file.readlines()

    start = START
    end = END

    for counter in range(1, STEP+1):
        inner_time_started = time.time()
        print("read::", str(counter), "th set")

        # Fetch and process articles asynchronously
        # Extract path from each article object and
        # make a request to read the article
        # clean the html body of the article

        cleaned_content = asyncio.run(fetch_articles(url_paths[start:end]))

        # Wait for a millisecond before making the next request
        time.sleep(REQUEST_WAIT_TIME)

        # Write into the file in an append mode or write mode if the loop counter
        # just started, writing into the same file
        mode = APPEND_MODE if counter > 1 else WRITE_MODE

        df = pd.DataFrame(cleaned_content)

        # Write the DataFrame to a CSV file
        df.to_csv(FILE_NAME_CSV, mode=mode, header=False,
                  index=False, encoding=FILE_ENCODING)

        # Since we are slicing, the start becomes the end
        # and the end becomes the number of data to process in a single loop
        # times the loop pointing variable, counter, as the loop counter increase
        # So if start = 0, and end = 10 and the END = 10, the next will be
        # start = end, which is 10, and end = 10 * (counter + 1)
        # end will ways be a factor of END then
        start = end
        end = END*(counter+1)

        inner_time_ended = time.time()
        inner_elapsed_time = inner_time_ended - inner_time_started
        print(
            f"Inner process time elapsed: {inner_elapsed_time:.2f} seconds\n")

    print(f"Data collected successfully, written into: {FILE_NAME_CSV}")
except Exception as e:
    print("An error occurred while collecting data")
    print(e)
finally:
    print("Done")
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time

    print(f"Total time elapsed: {elapsed_time:.2f} seconds")
    print(f"Fetched {END * STEP} DataSet")
