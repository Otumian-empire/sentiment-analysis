# from bs4 import BeautifulSoup
# import pandas as pd
# import numpy as np
# import re
# import nltk
# from nltk.corpus import stopwords
# from sklearn.model_selection import train_test_split
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.layers import Embedding, LSTM, Dense
# from keras.models import Sequential
# from keras.utils import to_categorical


# # Define the columns of the DataFrame
# # index, raw article, processed article, sentiment
# # int,   string,      string,            string
# # index, text,     processed_text, sentiment
# # negative = 0
# # neutral = 1
# # positive = 2

# # index is default
# DATA_FRAME_COLUMNS = ["index", "text", "processed_text", "sentiment"]
# NEGATIVE = 0
# NEUTRAL = 1
# POSITIVE = 2

# # Load article data set
# csv_df = pd.read_csv("./dev_articles.csv",
#                      encoding="utf-8", names=DATA_FRAME_COLUMNS, skiprows=1)

# # Print the shape and head of dataframe of the csv file
# print("Shape of training dataframe: ", csv_df.shape)
# print(csv_df.head())

# # Download stopwords
# nltk.download("stopwords")
# stop_words = set(stopwords.words("english"))


# # Create a SentimentIntensityAnalyzer instance
# SIA = SentimentIntensityAnalyzer()


# # Preprocess text data
# def preprocess_text(text):
#     if isinstance(text, str):
#         # Convert to lowercase
#         text = text.lower()

#         # Remove special characters and numbers
#         text = re.sub(r"[^a-zA-Z]", " ", text)

#         # Tokenization
#         words = text.split()

#         # Remove stopwords
#         words = [word for word in words if word not in stop_words]

#         # Join the words back to form a processed sentence
#         processed_text = " ".join(words)

#         return processed_text
#     else:
#         return ""  # Return an empty string for NaN values


# # compute the sentiment using a vader
# # update the implementation to use another method
# def compute_sentiment(text):
#     sentiment_scores = SIA.polarity_scores(text)
#     score = sentiment_scores["compound"]
#     # print("Text:", text)
#     # print("Sentiment Score:", score)

#     if score >= 0.05:
#         sentiment = POSITIVE
#     elif score <= -0.05:
#         sentiment = NEGATIVE
#     else:
#         sentiment = NEUTRAL

#     # print("Assigned Sentiment:", sentiment)
#     return sentiment


# # Apply preprocessing and compute sentiment for each row
# for index, row in csv_df.iterrows():
#     content = row["text"]
#     if isinstance(content, str):
#         processed_text = preprocess_text(content)
#         sentiment = int(compute_sentiment(content))

#         # print(index, "==>", sentiment)
#         if sentiment < 0:
#             sentiment = 0

#         csv_df.at[index, "processed_text"] = processed_text
#         csv_df.at[index, "sentiment"] = sentiment


# print("Clean data set")


# # remove any value out of bound
# for index, row in csv_df.iterrows():
#     if len(str(row["sentiment"])) > 1:
#         csv_df.at[index, "sentiment"] = sentiment

# print("further clean data for out of place sentiment values")

# # Now you can split the data into training and testing sets
# X = csv_df["processed_text"]  # Your feature (input) data
# y = csv_df["sentiment"]  # Your target (output) data


# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42)
# print("Split data")

# # print(y_test)
# # exit()
# # Map sentiment labels to integer values
# label_mapping = {NEGATIVE: 0, NEUTRAL: 1, POSITIVE: 2}
# y_train_mapped = y_train.map(label_mapping)
# y_test_mapped = y_test.map(label_mapping)
# print("Map sentiment labels to integer values")


# # Tokenization and Padding
# tokenizer = Tokenizer()

# # Filter out rows with missing values and replace with empty string
# X_train_filtered = X_train.fillna("")
# X_test_filtered = X_test.fillna("")

# y_test = y_test.fillna("")
# y_train = y_train.fillna("")

# # Fit tokenizer on filtered data
# tokenizer.fit_on_texts(X_train_filtered)
# X_train_sequences = tokenizer.texts_to_sequences(X_train_filtered)
# X_test_sequences = tokenizer.texts_to_sequences(X_test_filtered)
# X_train_padded = pad_sequences(X_train_sequences)
# X_test_padded = pad_sequences(X_test_sequences, maxlen=X_train_padded.shape[1])

# print("Tokenization and Padding")


# # Define Model Architecture
# vocabulary_size = len(tokenizer.word_index) + 1
# # Add 1 for padding token
# EMBEDDING_DIM = 100
# LONG_SHORT_TERM_MEMORY_LAYER = 128
# DENSE_UNIT = 3
# ACTIVATION = "softmax"

# model = Sequential([
#     Embedding(input_dim=vocabulary_size, output_dim=EMBEDDING_DIM,
#               input_length=X_train_padded.shape[1]),
#     LSTM(LONG_SHORT_TERM_MEMORY_LAYER, return_sequences=True),
#     LSTM(LONG_SHORT_TERM_MEMORY_LAYER),
#     Dense(3, activation=ACTIVATION)  # 3 classes: Negative, Neutral, Positive
# ])

# # model.compile(optimizer='adam',
# #               loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy', metrics=['accuracy'])


# print("Define Model Architecture")


# # Train the Model
# history = model.fit(X_train_padded, y_train_mapped, epochs=2,
#                     validation_split=0.1, batch_size=32)

# # history = model.fit(X_train_padded, y_train_mapped, epochs=2,
# #                     validation_split=0.1, batch_size=100)
# print("Train the Model")

# # Evaluate the Model:
# loss, accuracy = model.evaluate(X_test_padded, y_test)
# print("Test Loss:", loss)
# print("Test Accuracy:", accuracy)


# # Load test data
# test_data_df = pd.read_csv("./test_dev_articles.csv",
#                            encoding="utf-8",  skiprows=1)

# print("Read test data set")


# # Predict Sentiment
# for index, row in test_data_df[:1].iterrows():
#     new_article_sequence = tokenizer.texts_to_sequences(row[1])
#     new_article_padded = pad_sequences(
#         new_article_sequence, maxlen=X_train_padded.shape[1])

#     predictions = model.predict(new_article_padded)
#     print(predictions)

#     prediction_score = np.argmax(predictions)
#     print("prediction score:", prediction_score)

#     sentiment_labels = ["Negative", "Neutral", "Positive"]
#     predicted_sentiment = sentiment_labels[prediction_score]
#     print("Predicted Sentiment:", predicted_sentiment)
