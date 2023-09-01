# data_lexicon_vader.py

# Running sentimental analysis using Vader
# There is no cleaning needed here because the cleaning is already

import time
from matplotlib import pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import pandas as pd
import seaborn as sns
import nltk

# Download the vader lexicons
nltk.download("vader_lexicon")


# Load article data set
df = pd.read_csv("./preprocess_dev_articles.csv",
                 names=["processed_text"], encoding="utf-8")

# Shape of dataframe
print("Shape of training dataframe: ", df.shape)


# Using vader as a lexicon sentiment intensity analyzer
# VADER is a lexicon and rule-based feeling analysis instrument
#  that is explicitly sensitive to suppositions communicated in
#  web-based media.
analyzer = SIA()


def get_sentiment(content: str):
    return analyzer.polarity_scores(content)


# Analyze sentiment for each article
positive_count = 0
negative_count = 0
neutral_count = 0


# Start time
start_time = time.time()


# Loop through the data frame to determine the sentiments
for index, row in df.iterrows():
    # Column index 1 contains the article content
    # same as property, processed_text
    content = str(row["processed_text"])

    sentiment_scores = get_sentiment(content)
    print(f"✅️ applied Vader on item {index + 1}")

    compound_score = sentiment_scores["compound"]
    # if index > 4:
    #     exit()

    if compound_score >= 0.05:
        positive_count += 1
    elif compound_score <= -0.05:
        negative_count += 1
    else:
        neutral_count += 1

    # Print the sentiment scores for an article
    # print(f"Article {index + 1} - Sentiment Scores: {sentiment_scores}")


total_articles = df.shape[0]

print("Positive Articles:", positive_count)
print("Negative Articles:", negative_count)
print("Neutral Articles:", neutral_count)

print("Positive Percentage:", (positive_count / total_articles) * 100)
print("Negative Percentage:", (negative_count / total_articles) * 100)
print("Neutral Percentage:", (neutral_count / total_articles) * 100)

""" 
The output from these sentiment scores for each article, based on the VADER sentiment analysis, which provided the negativity, neutrality, positivity and compound values, demonstrated that even though some articles have strong positive or negative sentiment, others are more neutral. However, based on the compounds scores, we might conclude majority of these are sentimentally neutral.

If we tweak the code a bit and compute or count the sentiment variables based on the compound scores, we'd rather conclude that majority of the articles are positive.

NB: a compound score that combines the three previous scores

Positive Percentage: 91.5
Negative Percentage: 5.8
Neutral Percentage: 2.7
Time elapsed: 316.48s
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
plt.title("Sentiment Distribution of Articles using Vader")
plt.show()
