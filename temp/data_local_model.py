# # creating the model from ground up


# import tensorflow as tf
# import numpy as np
# from sklearn.metrics import accuracy_score
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.model_selection import train_test_split
# import re
# from nltk.corpus import stopwords
# import nltk
# import pandas as pd
# from gensim.models import Word2Vec


# # 1. Data Collection and Annotation
# # Gather a dataset of texts along with their corresponding sentiment labels.
# # These labels could be binary (positive/negative) or continuous
# # (sentiment score). You will need a sizable and diverse dataset for
# # training and evaluating your model.


# # Load article data set
# df = pd.read_csv("./dev_articles.csv", encoding="utf-8")

# # Shape of dataframe
# print(" Shape of training dataframe: ", df.shape)


# # 2. Text Preprocessing: Preprocess the text data to clean it and convert
# # it into a suitable format for analysis. This includes tasks like
# # lowercasing, removing punctuation, tokenization, stemming/lemmatization,
# # and more.


# # Download stopwords
# nltk.download("stopwords")
# stop_words = set(stopwords.words("english"))


# def preprocess_text(text):
#     if isinstance(text, str):  # Check if the value is a string
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


# # Create a list of data.
# # data = [[index, row[1], preprocess_text(row[1])]
# #         for index, row in df.iterrows()]

# data = []
# for index, row in df.iterrows():
#     print(index)
#     print(row)
#     content = row[1]

#     if content != np.NaN:
#         data.append([index, content, preprocess_text(content)])


# # Create a DataFrame for the processed data and apply the
# # preprocessing
# processed_text_df = pd.DataFrame(
#     data, columns=["index", "text", "processed_text"])

# # Print the DataFrame
# print(processed_text_df)


# # Train Word2Vec embeddings on your preprocessed text
# sentences = [text.split() for text in processed_text_df["processed_text"]]
# model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)


# # 3. Feature Extraction
# # Convert the processed text data into numerical features that machine learning
# # models can understand. Common techniques include TF-IDF (Term Frequency-Inverse
# # Document Frequency), word embeddings (Word2Vec, GloVe), and more aYes, you can
# # definitely create your own sentiment analysis module. Creating your own module
# # allows you to customize the sentiment analysis process according to your
# # specific requirements. Here's a basic outline of how you could go about creating
# # your own sentiment analysis module:

# # Using Term Frequency-Inverse Document Frequency to convert text data into
# # numerical vectors,

# # # Create a TF-IDF vectorizer
# # # You can adjust max_features as needed
# # vectorizer = TfidfVectorizer(max_features=2500)

# # # Fit and transform the processed content to TF-IDF features
# # transformed_features = vectorizer.fit_transform(
# #     processed_text_df["processed_text"])

# # # Convert the TF-IDF features to a dense array
# # # Create a DataFrame from the TF-IDF array
# # vectorized_df = pd.DataFrame(
# #     transformed_features.toarray(), columns=vectorizer.get_feature_names_out())

# # # Display the TF-IDF DataFrame
# # print(vectorized_df)

# # Get the embeddings for each word in your processed text
# word_embeddings = []
# for sentence in sentences:
#     embeddings = []
#     for word in sentence:
#         embeddings.append(model.wv[word])
#     word_embeddings.append(embeddings)

# # Convert the list of word embeddings into a DataFrame
# embedding_df = pd.DataFrame(word_embeddings)


# # 4. Model Selection and Training
# # Choose a suitable machine learning model for sentiment analysis.
# # This could be a traditional classifier like Naive Bayes or a more
# # complex model like a neural network. Split your dataset into
# # training and testing sets, and train your chosen model on the training data.


# # # Split the data into training and testing sets
# # X = vectorized_df  # Your TF-IDF features
# # y = processed_text_df["processed_text"]  # processed text

# # X_train, X_test, y_train, y_test = train_test_split(
# #     X, y, test_size=0.2, random_state=429)

# # # Initialize and train the Multinomial Naive Bayes classifier
# # # model = ComplementNB(force_alpha=True)
# # # model = BernoulliNB(force_alpha=True)
# # model = MultinomialNB(force_alpha=True)
# # model.fit(X_train, y_train)

# # # Predict sentiment labels for the testing set
# # y_pred = model.predict(X_test)

# # # Calculate accuracy
# # accuracy = accuracy_score(y_test, y_pred)
# # print("Accuracy:", accuracy)

# # Split the data into training and testing sets
# X = np.array(embedding_df)  # Use the word embeddings DataFrame
# y = np.array(processed_text_df["processed_text"])

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=429)

# # # Initialize and train the Multinomial Naive Bayes classifier
# # model = MultinomialNB(force_alpha=True)
# # model.fit(X_train, y_train)

# # Build the neural network model
# # model = Sequential([
# #     # Adjust input shape according to your Word2Vec dimensions
# #     Flatten(input_shape=(100,)),
# #     Dense(128, activation='relu'),
# #     # Use 'sigmoid' for binary sentiment classification
# #     Dense(1, activation='sigmoid')
# # ])

# # # Compile the model
# # model.compile(optimizer='adam', loss='binary_crossentropy',
# #               metrics=['accuracy'])

# # # Train the model
# # model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# # # Predict sentiment labels for the testing set
# # y_pred = model.predict(X_test)

# # y_pred_binary = np.round(y_pred)  # Convert probabilities to binary predictions
# # accuracy = accuracy_score(y_test, y_pred_binary)

# # # Calculate accuracy
# # accuracy = accuracy_score(y_test, y_pred)
# # print("Accuracy:", accuracy)


# # ... your previous code ...

# # Convert your NumPy arrays to TensorFlow tensors
# # print(X_train)
# # Convert your NumPy arrays to TensorFlow tensors
# # Convert your NumPy arrays to lists
# # Convert your lists to TensorFlow tensors

# def pad_list(list_to_pad, pad_value=0):
#     """Pad a list to a specified size."""
#     max_len = max(len(item) for item in list_to_pad)
#     return [
#         [float(item[0]) for item in list_item]
#         + [pad_value] * (max_len - len(list_item))
#         for list_item in list_to_pad
#     ]


# X_train_list = X_train.tolist()
# X_test_list = X_test.tolist()
# y_train_list = y_train.tolist()
# y_test_list = y_test.tolist()

# X_train_list_padded = pad_list(X_train_list)
# X_test_list_padded = pad_list(X_test_list)

# # Convert your lists to TensorFlow tensors
# X_train_tensor = tf.convert_to_tensor(
#     [item for sublist in X_train_list_padded for item in sublist],
#     dtype=tf.float32,
# )
# X_test_tensor = tf.convert_to_tensor(
#     [item for sublist in X_test_list_padded for item in sublist],
#     dtype=tf.float32,
# )
# y_train_tensor = tf.convert_to_tensor(
#     [item for sublist in y_train_list for item in sublist], dtype=tf.float32)
# y_test_tensor = tf.convert_to_tensor(
#     [item for sublist in y_test_list for item in sublist], dtype=tf.float32)


# # Initialize and train the Multinomial Naive Bayes classifier
# model = MultinomialNB(force_alpha=True)
# model.fit(X_train, y_train)

# # Predict sentiment labels for the testing set
# y_pred = model.predict(X_test)

# # Calculate accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)


# # 5. Model Evaluation: Evaluate the performance of your trained model using the testing dataset. Common evaluation metrics for sentiment analysis include accuracy, precision, recall, F1-score, and ROC-AUC.


# # 6. Module Integration: Once you have a trained model that can predict sentiment, you can encapsulate it in a Python module. This module should have a function or method that takes a text input and returns the sentiment prediction.


# # 7. Optional: Lexicon Integration: If you want to combine your model with lexicon-based analysis, you can integrate external sentiment lexicons like AFINN, SentiWordNet, etc. into your module's analysis process.


# # 8. Deployment: You can package your sentiment analysis module and deploy it as a library that others can use. You can also create a user-friendly interface, such as a web application, to interact with your module.
