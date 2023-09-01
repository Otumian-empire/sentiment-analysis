from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import sklearn.metrics as metrics
from keras.models import load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Activation, Dropout
from tensorflow.keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
from keras.utils import to_categorical
warnings.filterwarnings("ignore")


# Load data
DATA_FRAME_COLUMNS = ["index", "text", "processed_text", "sentiment"]
NEGATIVE = "negative"
NEUTRAL = "neutral"
POSITIVE = "positive"

# Load article data set
csv_df = pd.read_csv("./dev_articles.csv",
                     encoding="utf-8", names=DATA_FRAME_COLUMNS, skiprows=1)
print("Load article DataSet")

# csv_df.info()
print(csv_df.head())

# csv_df = csv_df[["processed_text", "sentiment"]]
# print(csv_df.head())


# Download stopwords
""" nltk.download("stopwords")"""
stop_words = set(stopwords.words("english"))


# Create a SentimentIntensityAnalyzer instance
SIA = SentimentIntensityAnalyzer()


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


# Apply preprocessing and compute sentiment for each row
for index, row in csv_df.iterrows():
    content = row["text"]
    if isinstance(content, str):
        processed_text = preprocess_text(content)
        sentiment = compute_sentiment(content)

        csv_df.at[index, "processed_text"] = processed_text
        csv_df.at[index, "sentiment"] = sentiment


print("Clean data set")
print(csv_df.head())


# Exploratory data analysis
# csv_df = csv_df[csv_df["sentiment"] != "positive"]
# print(csv_df.head())
# csv_df.info()


# Sentiment distribution
sentiment = csv_df["sentiment"].value_counts()
plt.figure(figsize=(12, 4))
sns.barplot(x=sentiment.index, y=sentiment.values, alpha=0.8)
plt.ylabel("Number of Occurrences", fontsize=12)
plt.xlabel("sentiment", fontsize=12)
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

#
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
print("Vocabulary size={}".format(len(tokenizer.word_index)))
print("Number of Documents={}".format(tokenizer.document_count))

# Sequence
X_train = tokenizer.texts_to_sequences([str(text) for text in X_train])
X_test = tokenizer.texts_to_sequences([str(text) for text in X_test])


train_lens = [len(s) for s in X_train]
test_lens = [len(s) for s in X_test]

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
h1 = ax[0].hist(train_lens)
h2 = ax[1].hist(test_lens)
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
ACTIVATION = "softmax"
vocabulary_size = len(tokenizer.word_index) + 1
learning_rate = 0.00001
model = Sequential()


model.add(Embedding(
    vocabulary, output_dim=EMBEDDING_DIM,
          input_length=MAX_SEQUENCE_LENGTH))

# model.add(Embedding(input_dim=vocabulary_size, output_dim=EMBEDDING_DIM,
#                     input_length=MAX_SEQUENCE_LENGTH))

# model.add(LSTM(LONG_SHORT_TERM_MEMORY_LAYER, return_sequences=True))
# model.add(LSTM(LONG_SHORT_TERM_MEMORY_LAYER))
model.add(Dense(3, activation=ACTIVATION))

# model.add(Conv1D(64, 8, activation="relu"))
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

early_stopping = EarlyStopping(
    monitor="val_loss", mode="min", verbose=VERBOSE,
    patience=NUMBER_OF_EPOCH_WITH_NO_IMPROVEMENTS)

model_checkpoint = ModelCheckpoint("./best_model/best_model_cnn1d.h5",
                                   monitor="val_accuracy", mode="max",
                                   verbose=VERBOSE, save_best_only=True)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    batch_size=BATCH_SIZE, shuffle=True,
                    validation_split=0.1, epochs=EPOCHS, verbose=VERBOSE,
                    callbacks=[early_stopping, model_checkpoint])


# Evaluation

# Model Accuracy
saved_model = load_model("./best_model/best_model_cnn1d.h5")
train_acc = saved_model.evaluate(X_train, y_train, verbose=VERBOSE)
test_acc = saved_model.evaluate(X_test, y_test, verbose=VERBOSE)
print(f"Train: {train_acc[1]*100}, Test: {test_acc[1]*100}")

# Identify Over fitting
# summarize history for accuracy
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
""" plt.show() """

# summarize history for loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
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


labels = [POSITIVE, NEGATIVE, NEUTRAL]
# print(classification_report(y_test, predictions(X_test)))
# exit()
# print("CNN 1D Accuracy: %.2f%%" %
#       (accuracy_score(y_test, predictions(X_test))*100))

# print("CNN 1D Precision: %.2f%%" %
#       (precision_score(y_test, predictions(X_test), average="macro")*100))

# print("CNN 1D Recall: %.2f%%" %
#       (recall_score(y_test, predictions(X_test), average="macro")*100))

# print("CNN 1D f1_score: %.2f%%" %
#       (f1_score(y_test, predictions(X_test), average="macro")*100))

print("================================================\n")
# print(classification_report(y_test, predictions(X_test)))


# exit()
# conf_matrix = confusion_matrix(y_test, predictions(X_test))
# print("Confusion Matrix Shape:", conf_matrix.shape)
# # pd.DataFrame(confusion_matrix(y_test, predictions(X_test)))
# confusion_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)
# print("Confusion Matrix:")
# print(confusion_df)


# """ # ROC AUC
# def plot_roc_curve(y_test, y_prediction):
#     fpr, tpr, _ = roc_curve(y_test, y_prediction, pos_label=2)
#     plt.plot(fpr, tpr)
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")


# plot_roc_curve(y_test, predictions(X_test))
# print("model AUC score: %.2f%%" %
#       (roc_auc_score(y_test, predictions(X_test), multi_class="ovr")*100))

# def plot_roc_curve(y_test, y_prediction):
#     fpr, tpr, _ = metrics.roc_curve(y_test, y_prediction, pos_label=2)
#     plt.plot(fpr, tpr)
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")


# Assuming your predictions are probabilities for the positive class
# y_prediction = model.predict(X_test)

# print(y_prediction.head())

# plot_roc_curve(y_test, y_prediction)
# print("model AUC score: %.2f%%" %
#       (metrics.roc_auc_score(y_test, y_prediction,  multi_class="ovr")))

y_pred_test = predictions(X_test)


# classification report
print("\n")
print(classification_report(y_test, y_pred_test))

# confusion matrix
confmat = confusion_matrix(y_test, y_pred_test)

fig, ax = plt.subplots(figsize=(4, 4))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)


for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va="center", ha="center")

plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.tight_layout()
plt.show()
