from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)

# Sample Parameters
num_questions = 10
embedding_dim = 32
rnn_units = 64

# Simulated data for model training (dummy data)
X_train = np.random.randint(0, num_questions, (500, 10))
y_train = np.random.randint(0, 2, (500, 10))

# Define the GRU-based model
model = Sequential([
    Embedding(input_dim=num_questions, output_dim=embedding_dim, input_length=10),
    GRU(rnn_units, return_sequences=False),
    Dense(num_questions, activation='sigmoid')
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    answers = np.array([data['question_ids']])
    prediction = model.predict(answers)
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(debug=True)
