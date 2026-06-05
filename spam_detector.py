import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =====================================================================
# STEP 1: Load the messages and labels (spam or ham)
# =====================================================================
print("--- Step 1: Loading Dataset ---")
# We use a reliable online public SMS Spam dataset for quick testing
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'text'])

# Let's peek at the first 5 rows of our data
print(df.head())
print(f"Total rows loaded: {len(df)}\n")


# =====================================================================
# STEP 2 & 3: Preprocess Text & Convert to Numeric Features (TF-IDF)
# =====================================================================
print("--- Steps 2 & 3: Preprocessing & Converting Text to Numbers ---")
# Convert labels from words into numbers: 'spam' becomes 1, 'ham' becomes 0
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

X = df['text']   # The actual text messages
y = df['label']  # The 0s and 1s labels

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TfidfVectorizer does all the heavy lifting automatically:
# 1. Lowercasing: Turns everything to lowercase.
# 2. Tokenization: Splits sentences into individual words.
# 3. Stopwords: Automatically drops useless common words like 'the', 'is', 'at'.
# 4. TF-IDF: Converts words into numeric importance scores.
tfidf_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)

# Train the vectorizer on our training text and transform it into math matrices
X_train_numeric = tfidf_vectorizer.fit_transform(X_train)
X_test_numeric = tfidf_vectorizer.transform(X_test)

print("Text successfully converted to numerical features!\n")


# =====================================================================
# STEP 4: Train a Simple Model (Naive Bayes)
# =====================================================================
print("--- Step 4: Training the Naive Bayes Classifier ---")
# Naive Bayes is an excellent, fast algorithm for text classification
model = MultinomialNB()
model.fit(X_train_numeric, y_train)
print("Model training complete.\n")


# =====================================================================
# STEP 5: Measure Performance
# =====================================================================
print("--- Step 5: Evaluating Performance Metrics ---")
# Make predictions on the hidden test set
predictions = model.predict(X_test_numeric)

# Calculate performance metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

# Print out the results clearly
print(f"Accuracy:  {accuracy:.4f} (Overall correct guesses)")
print(f"Precision: {precision:.4f} (Out of all items flagged as spam, how many actually were)")
print(f"Recall:    {recall:.4f} (Out of all real spam out there, how much did we catch)")
print(f"F1 Score:  {f1:.4f} (A balanced score combining precision and recall)\n")


# =====================================================================
# BONUS: Test It Yourself!
# =====================================================================
print("--- Bonus: Testing Custom Sentences ---")
custom_emails = [
    "Hey, are we still meeting for lunch at 1 PM today?",
    "CONGRATULATIONS! You won a $1,000 Walmart gift card. Click here to claim your cash prize now!"
]

# Convert custom text into the same numeric format
custom_numeric = tfidf_vectorizer.transform(custom_emails)
custom_predictions = model.predict(custom_numeric)

for email, pred in zip(custom_emails, custom_predictions):
    result = "SPAM" if pred == 1 else "HAM (Legitimate)"
    print(f"Email Text: '{email}' -> Result: {result}")