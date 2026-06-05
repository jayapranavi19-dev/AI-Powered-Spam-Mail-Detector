# AI-Powered-Spam-Mail-Detector
An intelligent, lightweight Natural Language Processing (NLP) application built in Python that accurately distinguishes between spam and legitimate (ham) messages using machine learning. This repository serves as a practical introduction to text classification, taking raw text data, cleaning it, and running it through a statistical classifier.
Features:
Full Text Preprocessing: Automates lowercasing, word tokenization, and stop-word removal.
Vectorization: Converts raw text into meaningful numerical data using TF-IDF (Term Frequency-Inverse Document Frequency) weights.
Machine Learning Brain: Utilizes a highly efficient Multinomial Naive Bayes classifier ideal for text tasks.
Performance Metrics: Evaluates success using real-world metrics: Accuracy, Precision, Recall, and F1-Score.
Live Demo Script: Includes custom test cases to see the trained model classify brand-new sentences in real-time.
This project is an AI-powered text classifier designed to automatically detect and filter out spam messages. Built using Python, it takes raw, messy text data and utilizes Natural Language Processing (NLP) to clean, lowercase, and tokenize sentences into key vocabulary. The text is then translated into mathematical values using TF-IDF vectorization so the computer can understand word importance. A Multinomial Naive Bayes machine learning model is trained on this data to calculate the probability of a message being junk based on its phrasing. The final system achieves a high accuracy of over 97%, effectively protecting user inboxes while ensuring important, legitimate emails are never lost.
