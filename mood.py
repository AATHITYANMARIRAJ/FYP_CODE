import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
import nltk
from nltk.corpus import stopwords
import re

# Ensure stopwords are downloaded for NLTK
nltk.download('stopwords')

# 1. Data Loading and Preprocessing
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.strip()  # Remove whitespaces
    text = " ".join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Load dataset (replace 'dataset.csv' with your file path)
data = pd.read_csv('dataset.csv')
data['clean_text'] = data['forum_posts'].apply(preprocess_text)

# Extract main features
features = data[['time_spent', 'posts_count', 'quizzes_attempted', 'scores', 'forum_participation']]
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Anomaly Detection
iso = IsolationForest(contamination=0.1)
data['anomaly'] = iso.fit_predict(features_scaled)
data = data[data['anomaly'] == 1]

# 2. Emotion Detection Model with BiLSTM
# Tokenization and padding
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data['clean_text'])
X_text = tokenizer.texts_to_sequences(data['clean_text'])
X_text = pad_sequences(X_text, maxlen=100)

# BiLSTM model for emotion classification
emotion_model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=100),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(32)),
    Dense(6, activation='softmax')  # Assuming six emotion classes
])
emotion_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Placeholder for model training
# emotion_model.fit(X_text, y_emotion, epochs=5, batch_size=32, validation_split=0.2)

# 3. Clustering Learners (K-means)
# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

kmeans = KMeans(n_clusters=3, random_state=42)
data['engagement_cluster'] = kmeans.fit_predict(features_pca)

# Map clusters to engagement levels
engagement_map = {0: 'active', 1: 'passive', 2: 'observer'}
data['engagement_level'] = data['engagement_cluster'].map(engagement_map)

# 4. Decision Tree Classifier for Engagement Prediction
X = features_scaled
y = data['engagement_level'].map({'active': 2, 'passive': 1, 'observer': 0})  # Encode labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree Classifier
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)

# Evaluate the classifier
y_pred = dt_clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, dt_clf.predict_proba(X_test), multi_class="ovr"))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Cross-validation scores
cross_val = cross_val_score(dt_clf, X, y, cv=10)
print("Cross-Validation Accuracy:", cross_val.mean())

# 5. Optional: Visualization
import matplotlib.pyplot as plt

# PCA Plot of Engagement Levels
plt.scatter(features_pca[:, 0], features_pca[:, 1], c=data['engagement_cluster'], cmap='viridis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-means Clustering of Learners')
plt.colorbar()
plt.show()
