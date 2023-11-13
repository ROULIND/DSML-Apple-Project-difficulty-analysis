

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Load the datasets
train_data = pd.read_csv("training_data.csv")
test_data = pd.read_csv("unlabelled_test_data.csv")

# Preprocess the data (e.g., text cleaning, handling missing values)
# Your preprocessing steps go here



# Example with TF-IDF
tfidf = TfidfVectorizer(max_features=1000)  # You can adjust the number of features
train_vectors = tfidf.fit_transform(train_data['text'])
test_vectors = tfidf.transform(test_data['text'])


model = RandomForestClassifier()
scores = cross_val_score(model, train_vectors, train_data['difficulty'], cv=5, scoring='accuracy')
print("Average Cross-Validation Accuracy:", scores.mean())

# Training the final model
model.fit(train_vectors, train_data['difficulty'])

# Making predictions
predictions = model.predict(test_vectors)

# Preparing the submission file
submission = pd.DataFrame({'id': test_data['id'], 'difficulty': predictions})
submission.to_csv('/mnt/data/my_submission.csv', index=False)
