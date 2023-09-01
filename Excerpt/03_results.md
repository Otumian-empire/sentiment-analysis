# Result

## VADER Sentiment Analysis:

- **Positive Percentage:** 91.5%
- **Negative Percentage:** 5.8%
- **Neutral Percentage:** 2.7%
- **Time Elapsed:** 316.48s

VADER predominantly categorized articles as positive, with a smaller proportion being negative, and very few as neutral. The analysis was relatively fast, taking around 316.48 seconds.

## TextBlob Sentiment Analysis:

- **Positive Percentage:** 85.8%
- **Negative Percentage:** 12.1%
- **Neutral Percentage:** 2.1%
- **Time Elapsed:** 149.00s

TextBlob identified a majority of articles as positive, followed by a significant number of negative articles, and a very small percentage as neutral. The analysis was completed in approximately 149.00 seconds.

## SentiWordNet Sentiment Analysis:

- **Positive Percentage:** 95.4%
- **Negative Percentage:** 4.3%
- **Neutral Percentage:** 0.3%
- **Time Elapsed:** 1613.69s

SentiWordNet identified a large portion of articles as positive, with a much smaller percentage as negative and an extremely low number as neutral. The analysis was the slowest, taking approximately 1613.69 seconds.

## Affective Norms for English Words:

- **Positive Percentage:** 87.7%
- **Negative Percentage:** 5.4%
- **Neutral Percentage:** 6.9%
- **Time Elapsed:** 12.33s

Affective Norms for English Words identified a significant portion of articles as positive, with a few negative articles and some as neutral. This analysis was the fastest, completing in approximately 12.33 seconds.

## CNN Sentiment Analysis:

- **Accuracy:** 3%
- **Precision, Recall, and F1-score:** Very low values
- **Time Elapsed:** 510.12s

The CNN model had a very low accuracy, indicating that it struggled to classify the articles effectively. The precision, recall, and F1-score were also very low. The analysis took approximately 510.12 seconds.

These results provide insights into the effectiveness and efficiency of each sentiment analysis method for technical articles. It's clear that lexicon-based methods like VADER, TextBlob, and Affective Norms performed better in this context compared to the deep learning approach using CNNs, which had poor performance and was computationally expensive.
