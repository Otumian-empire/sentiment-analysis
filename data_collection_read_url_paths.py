# data_collection_read_url_paths.py

# Make API request to pull X number of article paths
# X is determined by the product of ARTICLES_PER_PAGE and API_CALLS
# The article paths will be used to pull and clean the
# article content itself
# python data_collection_read_url_paths.py


# Data collection from dev.to
import time
import requests
import os
from dotenv import load_dotenv


# load data from .env file
load_dotenv()


# Request constants
API_KEY = os.environ.get("API_KEY")

if API_KEY == None or len(API_KEY) < 1:
    print("Please create a `.env` file in the root of", end="")
    print("the application and add the `API_KEY`")
    exit()


# Base api url
DEV_TO_URL = "https://dev.to/api/articles"

# number of article per request, default = 30
ARTICLES_PER_PAGE = 1000

# Factor to multiply ARTICLES_PER_PAGE
# So if ARTICLES_PER_PAGE = 10, and API_CALLS = 2
# would have 2 * 10 article url paths
API_CALLS = 100

# File constants: where the url paths are save
FILE_NAME_URL_PATHS = "article_paths.txt"

# Request wait time
REQUEST_WAIT_TIME = 1
# so that the request won't DOS the server or we  be flagged as DOSing


try:
    article_paths = []

    # Make API_CALLS api calls
    for i in range(API_CALLS):

        # Make API request to fetch article paths
        response = requests.get(DEV_TO_URL, params={
            "per_page": ARTICLES_PER_PAGE,
            "api_key": API_KEY
        })

        # On success
        if response.status_code == 200:
            print("API call was successful")

            response_data = response.json()

            for index, article in enumerate(response_data):
                article_paths.append(article['path'])

            time.sleep(REQUEST_WAIT_TIME)
            print(f"Making {i+1}th request")

    print(f"There {len(article_paths)} articles read")
    print("\n".join(article_paths), file=open(FILE_NAME_URL_PATHS, "+w"))
    print(
        f"Fetched {ARTICLES_PER_PAGE * API_CALLS} article urls.", end="\n")

except Exception as e:
    print(e)
    print("An error occurred while collecting data")


# The break down makes the replication and maintenance of each component
# easier to implement, maintain and update
