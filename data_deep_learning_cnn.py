# data_deep_learning_cnn.py

# Running sentimental analysis using Deep learning CNN
# Where the sentiment are assigned with Vader
# There is no cleaning needed here because the cleaning is already


from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Embedding, Dropout
from sklearn.metrics import confusion_matrix, classification_report
import time
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import load_model
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import warnings
warnings.filterwarnings("ignore")


# Download stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


# Load data
DATA_FRAME_COLUMNS = ["processed_text", "sentiment"]
NEGATIVE = "negative"
NEUTRAL = "neutral"
POSITIVE = "positive"


# File constants
FILE_NAME = "./preprocess_dev_articles.csv"
FILE_ENCODING = "utf-8"
CSV_FIELD_NAMES = ["processed_text",]


# Start time
start_time = time.time()


# Load article data set
csv_df = pd.read_csv(FILE_NAME, names=CSV_FIELD_NAMES, encoding=FILE_ENCODING)

# Shape of dataframe
print("Shape of training dataframe: ", csv_df.shape)

# csv_df.info()
print(csv_df.head())

# csv_df = csv_df[["processed_text", "sentiment"]]
# print(csv_df.head())


# Create a SentimentIntensityAnalyzer instance
SIA = SentimentIntensityAnalyzer()


# compute the sentiment using a vader
# update the implementation to use another method
def compute_sentiment(text):
    sentiment_scores = SIA.polarity_scores(text)
    score = sentiment_scores["compound"]
    # print("Text:", text)
    # print("Sentiment Score:", score)

    if score >= 0.05:
        sentiment = POSITIVE
    elif score <= -0.05:
        sentiment = NEGATIVE
    else:
        sentiment = NEUTRAL

    # print("Assigned Sentiment:", sentiment)
    return sentiment


# Compute sentiment for each row
for index, row in csv_df.iterrows():
    content = str(row["processed_text"])
    sentiment = compute_sentiment(content)
    print(f"✅️ applied Vader on item {index + 1}")

    csv_df.at[index, "sentiment"] = sentiment


print("Clean data set")
print(csv_df.head())


# Exploratory data analysis
# csv_df = csv_df[csv_df["sentiment"] != "positive"]
print(csv_df.head())
# csv_df.info()


# Sentiment distribution
sentiment = csv_df["sentiment"].value_counts()
plt.figure(figsize=(12, 4))
sns.barplot(x=sentiment.index, y=sentiment.values, alpha=0.8)
plt.ylabel("Number of occurrences", fontsize=12)
plt.xlabel("sentiment", fontsize=12)
plt.title("Sentiment distribution")
# plt.xticks(rotation=90)  # do not rotate
""" plt.show() """


# Data preparation

# Data cleaning # is done in the reading pf the data frame


# Handling imbalance (oversampling)
# Separate majority and minority classes in training data for upsampling
data_majority = csv_df[csv_df["sentiment"] == "positive"]
data_minority = csv_df[csv_df["sentiment"] == "negative"]

print("majority class before upsample:", data_majority.shape)
print("minority class before upsample:", data_minority.shape)


# Upsample minority class
data_minority_upsampled = resample(data_minority,
                                   replace=True,     # sample with replacement
                                   # to match majority class
                                   n_samples=data_majority.shape[0],
                                   random_state=54)  # reproducible results

# Combine majority class with upsampled minority class
df_balance = pd.concat([data_majority, data_minority_upsampled])

# Display new class counts
print("After upsampling\n", df_balance["sentiment"].value_counts(), sep="")


# Data splitting
X_train, X_test, y_train, y_test = train_test_split(
    csv_df["processed_text"], csv_df["sentiment"], test_size=0.2, random_state=64)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Replace NA with empty string
X_train_filtered = X_train.fillna("")
X_test_filtered = X_test.fillna("")

y_test = y_test.fillna("")
y_train = y_train.fillna("")


# Tokenizer
tokenizer = Tokenizer()

# Convert NaN values to empty strings
X_train = X_train.fillna("")
X_train = X_train.astype(str)
tokenizer.fit_on_texts(X_train)

# Convert all text data to strings
# tokenizer.fit_on_texts([str(text) for text in X_train])

vocabulary = len(tokenizer.index_word) + 1
print(f"Vocabulary size={len(tokenizer.word_index)}")
print(f"Number of Documents={tokenizer.document_count}")

# Sequence
X_train = tokenizer.texts_to_sequences([str(text) for text in X_train])
X_test = tokenizer.texts_to_sequences([str(text) for text in X_test])


# Distribution of sequence lengths in the training and testing datasets
train_lens = [len(s) for s in X_train]
test_lens = [len(s) for s in X_test]

figure, axes = plt.subplots(1, 2, figsize=(12, 6))

# histogram for the training set
axes[0].hist(train_lens)

# histogram for the training set
axes[1].hist(test_lens)
plt.title("Distribution of sequence lengths in the training and testing datasets")
""" plt.show() """

# padding
MAX_SEQUENCE_LENGTH = 30
X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH, padding="post")
X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH, padding="post")
print(X_train.shape, X_test.shape)


# Encoding Labels
# let positive = 1, negative = 0
label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)


# Modelling

# Build model
VOCABULARY_SIZE = 1000
EMBEDDING_DIM = 100
LONG_SHORT_TERM_MEMORY_LAYER = 128
DENSE_UNIT = 3


vocabulary_size = len(tokenizer.word_index) + 1
learning_rate = 0.00001

model = Sequential()
model.add(Embedding(
    vocabulary, output_dim=EMBEDDING_DIM,
          input_length=MAX_SEQUENCE_LENGTH))

model.add(Dense(3, activation="softmax"))

model.add(Conv1D(filters=256, kernel_size=15, activation="relu"))
model.add(MaxPooling1D(2))
model.add(Dropout(0.3))

model.add(Dense(16, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(16, activation="relu"))
model.add(Dropout(0.3))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation="sigmoid"))  # Change this to sigmoid

model.compile(loss="binary_crossentropy", optimizer=tf.optimizers.Adam(
    learning_rate=learning_rate), metrics=["accuracy"])
model.summary()


# Train model
EPOCHS = 10
BATCH_SIZE = 256
NUMBER_OF_EPOCH_WITH_NO_IMPROVEMENTS = 25

VERBOSE = 1

MODEL_PATH = "./deep_model/ccn_model_cnn.h5"

early_stopping = EarlyStopping(
    monitor="val_loss", mode="min", verbose=VERBOSE,
    patience=NUMBER_OF_EPOCH_WITH_NO_IMPROVEMENTS)

model_checkpoint = ModelCheckpoint(MODEL_PATH,
                                   monitor="val_accuracy", mode="max",
                                   verbose=VERBOSE, save_best_only=True)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    batch_size=BATCH_SIZE, shuffle=True,
                    validation_split=0.1, epochs=EPOCHS, verbose=VERBOSE,
                    callbacks=[early_stopping, model_checkpoint])


# Evaluation

# Load the mode for a test
saved_model = load_model(MODEL_PATH)

train_accuracy = saved_model.evaluate(X_train, y_train, verbose=VERBOSE)
test_accuracy = saved_model.evaluate(X_test, y_test, verbose=VERBOSE)


print(
    f"Train: {(train_accuracy[1]*100):.2f}, Test: {(test_accuracy[1]*100):.2f}")


# end
end_time = time.time()
print(f"Time elapsed: {(end_time-start_time):.2f}s")


# Identify Over fitting
# summarize history for accuracy
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.title("Summarize History For Accuracy")

""" plt.show() """

# summarize history for loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.title("Summarize History For Loss")
""" plt.show() """


POSITIVE_SCORE = 1
NEGATIVE_SCORE = 0
THRESHOLD_SCORE = 0.5


# Confusion Matrix
def predictions(x):
    prediction_probability = model.predict(x)
    predictions = []

    for score in prediction_probability:
        if score > THRESHOLD_SCORE:
            predictions.append(POSITIVE_SCORE)
        else:
            predictions.append(NEGATIVE_SCORE)

    return predictions


print("\nClassification Report")
y_prediction_test = predictions(X_test)

# classification report
print("\n")
print(classification_report(y_test, y_prediction_test))


# confusion matrix
confusion_matrix_accuracy = confusion_matrix(y_test, y_prediction_test)


# Plot and how matrix heatmap
figure, axes = plt.subplots(figsize=(4, 4))
axes.matshow(confusion_matrix_accuracy, cmap=plt.cm.Blues, alpha=0.3)


for true_label in range(confusion_matrix_accuracy.shape[0]):
    for predicted_label in range(confusion_matrix_accuracy.shape[1]):
        axes.text(
            x=predicted_label, y=true_label,
            s=confusion_matrix_accuracy[true_label, predicted_label],
            va="center", ha="center"
        )


plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.tight_layout()
plt.title("Confusion Matrix: Model Predictions vs. Actual Labels")
plt.show()


# Testing on other data

TEST_ARTICLE_PATH = "test_dev_articles.csv"

# Load the trained model
saved_model = load_model(MODEL_PATH)

# Load test articles
test_df = pd.read_csv(TEST_ARTICLE_PATH,
                      names=["text"], encoding="utf-8", skiprows=1)


# Preprocess and predict sentiment for each testing article
for index, row in test_df.iterrows():
    article_text_processed = tokenizer.texts_to_sequences([row["text"]])
    article_text_processed = pad_sequences(
        article_text_processed, maxlen=MAX_SEQUENCE_LENGTH, padding="post")

    # Predict the sentiment
    prediction_probability = saved_model.predict(article_text_processed)

    if prediction_probability > THRESHOLD_SCORE:
        predicted_sentiment = "positive"
    else:
        predicted_sentiment = "negative"

    print("Article Text:", index+1)
    print("Predicted Sentiment:", predicted_sentiment)
    print("Vader Sentiment:", compute_sentiment(row["text"]))
    print()  # Print a blank line for separation


"""
Time elapsed: 510.12s
 

              precision    recall  f1-score   support

           0       0.00      0.00      0.00      1186
           1       0.03      1.00      0.05       530
           2       0.00      0.00      0.00     18284

    accuracy                           0.03     20000
   macro avg       0.01      0.33      0.02     20000
weighted avg       0.00      0.03      0.00     20000

"""
