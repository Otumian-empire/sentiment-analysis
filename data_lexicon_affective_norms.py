# data_lexicon_affective_norms.py

# Running sentimental analysis using Using Affective Norms for English Words
# There is no cleaning needed here because the cleaning is already

import time
from matplotlib import pyplot as plt
import pandas as pd
import pandas as pd
import seaborn as sns


# Load article DataSet
df = pd.read_csv("./preprocess_dev_articles.csv",
                 names=["processed_text"], encoding="utf-8")

# Shape of dataframe
print("Shape of training dataframe: ", df.shape)


# Using Affective Norms for English Words, as a lexicon sentiment analyzer
# Affective Norms for English Words assigns pre-computed sentiment scores
# to English words. The sentiment scores range from -5 to +5,
# where negative scores indicate negative sentiment,
# positive scores indicate positive sentiment,
# and a score of 0 implies neutral sentiment for that word.


# Load the Affective Norms for English Words
# The file was download from the web
# https://raw.githubusercontent.com/fnielsen/afinn/master/afinn/data/AFINN-111.txt
# and renamed it as - affective_norms_for_english_words.txt
with open("affective_norms_for_english_words.txt", 'r') as anew_file_obj:
    words = anew_file_obj.readlines()
    ANEW_ENGLISH_WORDS = dict()

    for line in words:
        word, score = line.split("\t")
        ANEW_ENGLISH_WORDS[word] = int(score)


def get_sentiment(text):
    words = text.split()
    sentiment_score = sum([ANEW_ENGLISH_WORDS.get(word, 0) for word in words])
    return sentiment_score


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
    print(f"✅️ applied ANEW on item {index + 1}")

    # Print the sentiment of each article
    # print(sentiment_score)

    # compute the sentiment percentage
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
The output from these sentiment scores for each article, based on the Affective Norms for English Words, which provided a value ranging between, -5.0 to 5.0 in a manner, spread between: negativity, neutrality, positivity, demonstrated that sentimentality is quite further from neutral when the values a inspected. However, using the the numeric range of indication, there are few neutral article. There were few negative articles and the rest were positive.


Positive Percentage: 87.7
Negative Percentage: 5.4
Neutral Percentage: 6.9
Time elapsed: 12.33s
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
plt.title(
    "Sentiment Distribution of Articles using Affective Norms for English Words")
plt.show()
