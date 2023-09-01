# data_lexicon_text_blob.py

# Running sentimental analysis using TextBlob
# There is no cleaning needed here because the cleaning is already

import time
from textblob import TextBlob
import pandas as pd
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


# Load article data set
df = pd.read_csv("./preprocess_dev_articles.csv",
                 names=["processed_text"], encoding="utf-8")

# Shape of dataframe
print("Shape of training dataframe: ", df.shape)

# Using TextBlob as a lexicon sentiment analyzer
# TextBlob is a textual data processor, which provides a consistent API
# for diving into common natural language processing (NLP) tasks such
#  as part-of-speech tagging, noun phrase extraction, sentiment analysis,
#  and more.


def get_sentiment(content: str):
    analysis = TextBlob(content)
    return analysis.sentiment.polarity


# Analyze sentiment for each article
positive_count = 0
negative_count = 0
neutral_count = 0


# start time
start_time = time.time()


# Loop through the data frame to determine the sentiments
for index, row in df.iterrows():
    # Column index 1 contains the article content
    # same as property, processed_text
    content = str(row["processed_text"])

    sentiment_score = get_sentiment(content)
    print(f"✅️ applied TextBlob on item {index + 1}")

    # Print the sentiment of each article
    # print(sentiment_score)
    # The sentiment property, polarity, refers to the emotional tone
    # of the text. It is typically measured on a numeric scale that
    # ranges from -1.0 to 1.0, where a polarity score greater than 0
    # indicates positive sentiment (e.g., happiness, satisfaction),
    # a polarity score less than 0 indicates negative sentiment
    # (e.g., sadness, frustration) and a polarity score around 0
    # indicates neutral sentiment (e.g., factual information).

    if sentiment_score > 0:
        positive_count += 1
    elif sentiment_score < 0:
        negative_count += 1
    else:
        neutral_count += 1

    # Print the sentiment score for the article
    # print(f"Article {index + 1} - Sentiment Scores: {sentiment_score}")


total_articles = df.shape[0]

print("Positive Articles:", positive_count)
print("Negative Articles:", negative_count)
print("Neutral Articles:", neutral_count)

print("Positive Percentage:", (positive_count / total_articles) * 100)
print("Negative Percentage:", (negative_count / total_articles) * 100)
print("Neutral Percentage:", (neutral_count / total_articles) * 100)

""" 
The output from these sentiment scores for each article, based on the TextBlob sentiment analysis, which provided a value ranging between, -1.0 to 1.0 in a manner, spread between: negativity, neutrality, positivity, demonstrated that sentimentality is quite closer to neutral when the values a inspected. However, using the the numeric range of indication, there is no specific neutral article. There was few negative articles and the rest were positive. 


Positive Percentage: 85.8
Negative Percentage: 12.1
Neutral Percentage: 2.1
Time elapsed: 149.00s
"""


# Create a DataFrame for the sentiment distribution
sentiment_labels = ["Negative", "Neutral", "Positive"]
sentiment_counts = [(negative_count / total_articles) * 100,
                    (neutral_count / total_articles) * 100,
                    (positive_count / total_articles) * 100]

sentiment_data = pd.DataFrame(
    {"Sentiment": sentiment_labels, "Count": sentiment_counts})


# end
end_time = time.time()
print(f"Time elapsed: {(end_time-start_time):.2f}s")


# Create a stacked bar chart
sns.set(style="whitegrid")
sns.barplot(x="Sentiment", y="Count", data=sentiment_data,
            palette=["green", "red", "blue"])
plt.xlabel("Sentiment")
plt.ylabel("Percentage")
plt.title("Sentiment Distribution of Articles using TextBlob")
plt.show()
