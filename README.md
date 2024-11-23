# Sentiment Analysis with Deep Learning and Naive Bayes

## Overview

This project aims to classify the sentiment of tweets using NLP Techniques,deep learning and Naive Bayes models. The dataset used is "tweet_emotions.csv", which contains tweets labeled with different sentiments. The project focuses on classifying tweets into four primary sentiments: worry, sadness, surprise, and love. 

## Steps

### 1. Data Loading and Preprocessing

- The code starts by importing necessary libraries like nltk, pandas, numpy, and re.
- It loads the tweet dataset using pandas and performs preprocessing steps:
    - Converting text to lowercase.
    - Removing unwanted sentiments (anger, neutral, happiness, fun, enthusiasm, boredom, empty, hate, relief).
    - Resetting the index.
- It then creates a list of sentiments, assigning numerical labels to each sentiment category (worry: 0, sadness: 1, surprise: 2, love: 3).
- Extracts tweets for each target sentiment (sadness, surprise, worry, love).
- Creates a list of all tweets for further processing.

### 2. Text Cleaning and Tokenization

- Imports necessary libraries for text cleaning and tokenization, including nltk.corpus, nltk.stem, and nltk.tokenize.
- Defines a `process_tweet` function to clean and tokenize tweets:
    - Removes stop words, punctuation, and applies stemming using PorterStemmer.
    - Cleans tweets by removing URLs, hashtags, and mentions.
- Defines a `count_tweets` function to count the frequency of words associated with each sentiment.

### 3. Feature Extraction

- Creates a dictionary (`pair_dic`) to store word-sentiment pairs and their frequencies using the `count_tweets` function.

### 4. Data Splitting

- Splits the data into training and testing sets (80% for training, 20% for testing).

### 5. Model Building and Training

**Deep Learning Model (LSTM):**
- Uses TensorFlow and Keras to build a Bidirectional LSTM model for sentiment classification.
- Tokenizes the text data using `Tokenizer`.
- Pads the sequences to a fixed length using `pad_sequences`.
- Compiles the model with `sparse_categorical_crossentropy` loss and `Adam` optimizer.
- Trains the model using `model.fit`.

**Naive Bayes Model:**
- Uses scikit-learn's `MultinomialNB` to build a Naive Bayes classifier.
- Trains the classifier using the training data.

### 6. Evaluation

- Evaluates the performance of the models using accuracy and classification report metrics.
- Predicts sentiment on sample sentences using the trained models.

## How to Run

1. Make sure you have the necessary libraries installed: nltk, pandas, numpy, re, tensorflow, scikit-learn. You can install them using `pip install library_name`.
2. Upload the `tweet_emotions.csv` dataset to your Google Colab environment.
3. Run the code cells in the provided notebook sequentially.

## Results

The results will show the accuracy and classification report for both the deep learning and Naive Bayes models. Additionally, the predicted sentiment for the sample sentences will be displayed.

## Conclusion

This project demonstrates how to build sentiment analysis models using deep learning and Naive Bayes techniques. By experimenting with different model architectures and hyperparameters, you can further improve the accuracy of sentiment classification.
