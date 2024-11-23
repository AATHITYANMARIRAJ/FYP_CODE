import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding
import tkinter as tk
from tkinter import messagebox

# Sample Parameters
num_questions = 100  # e.g., 100 different questions
embedding_dim = 32   # Dimension for embedding layers
rnn_units = 64       # RNN units for GRU layer

# Dummy Data for Model Training (Simulating questions and correctness)
X_train = np.random.randint(0, num_questions, (500, 10))  # Example question ids for 500 students
y_train = np.random.randint(0, 2, (500, 10))              # Binary correctness for each question

# Define a simple RNN model
model = Sequential([
    Embedding(input_dim=num_questions, output_dim=embedding_dim, input_length=10),
    GRU(rnn_units, return_sequences=True),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)  # Training the model on dummy data

# Tkinter GUI Application
class QuizApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Learning Capability Quiz")
        
        # Variables
        self.question_index = 0
        self.score = 0
        self.question_ids = []
        self.correct_answers = []

        # Dummy question bank (for simplicity, using a list)
        self.questions = [
            {"id": 1, "text": "What particles are found in the nucleus of an atom?", "answer": "Protons and Neutrons", "options": ["Electrons and Protons", "Protons and Neutrons", "Neutrons and Electrons", "Protons only"]},
            {"id": 2, "text": "Which force binds protons and neutrons in the nucleus?", "answer": "Nuclear force", "options": ["Gravitational force", "Electromagnetic force", "Nuclear force", "Weak force"]},
            {"id": 3, "text": "What is the process of splitting a heavy nucleus into two lighter nuclei called?", "answer": "Nuclear fission", "options": ["Nuclear fusion", "Nuclear fission", "Radioactive decay", "Beta decay"]},
            {"id": 4, "text": "What type of radiation consists of helium nuclei?", "answer": "Alpha radiation", "options": ["Beta radiation", "Gamma radiation", "Alpha radiation", "Neutron radiation"]},
            {"id": 5, "text": "Which of the following is true for radioactive decay?", "answer": "It is a random and spontaneous process", "options": ["It is influenced by temperature", "It is a random and spontaneous process", "It can be controlled by magnetic fields", "It is caused by chemical reactions"]},
            {"id": 6, "text": "The mass defect of a nucleus is related to its:", "answer": "Binding energy", "options": ["Charge", "Density", "Binding energy", "Half-life"]},
            {"id": 7, "text": "What is the time required for half of the radioactive nuclei in a sample to decay called?", "answer": "Half-life", "options": ["Decay constant", "Lifetime", "Half-life", "Average life"]},
            {"id": 8, "text": "Fusion reactions typically occur at:", "answer": "High temperature and high pressure", "options": ["Low temperature and high pressure", "High temperature and high pressure", "Low temperature and low pressure", "High temperature and low pressure"]},
            {"id": 9, "text": "Beta decay results in the emission of:", "answer": "Electrons or positrons", "options": ["Alpha particles", "Gamma rays", "Electrons or positrons", "Neutrons"]},
            {"id": 10, "text": "In a nuclear reactor, control rods are used to:", "answer": "Absorb neutrons", "options": ["Increase reaction rate", "Absorb neutrons", "Reflect neutrons", "Increase fuel temperature"]}
        ]
        
        # Question Label
        self.question_label = tk.Label(root, text="", font=("Arial", 14))
        self.question_label.pack(pady=20)

        # Answer Options
        self.options = tk.StringVar()
        self.option_buttons = []
        for i in range(4):
            btn = tk.Radiobutton(root, text="", variable=self.options, value=i, font=("Arial", 12))
            btn.pack(anchor="w")
            self.option_buttons.append(btn)

        # Submit Button
        self.submit_button = tk.Button(root, text="Submit Answer", command=self.submit_answer)
        self.submit_button.pack(pady=20)

        # Start Quiz
        self.next_question()

    def next_question(self):
        if self.question_index < len(self.questions):
            # Load the next question
            question = self.questions[self.question_index]
            self.question_label.config(text=question["text"])
            self.question_ids.append(question["id"])

            # Set options
            for i, option in enumerate(question["options"]):
                self.option_buttons[i].config(text=option, value=option)
            
            self.options.set("")  # Clear previous selection
        else:
            # Quiz is over
            self.display_results()

    def submit_answer(self):
        selected_option = self.options.get()
        if selected_option:
            question = self.questions[self.question_index]
            correct = selected_option == question["answer"]

            # Record if answer was correct
            self.correct_answers.append(int(correct))
            if correct:
                self.score += 1

            # Move to the next question
            self.question_index += 1
            self.next_question()
        else:
            messagebox.showwarning("Warning", "Please select an answer.")

    def display_results(self):
        # Get model predictions for learning classification
        inputs = np.array([self.question_ids])
        predictions = model.predict(inputs)
        avg_correctness = np.mean(self.correct_answers)

        # Classify based on the model's predictions and average correctness
        learner_type = self.classify_student(avg_correctness)
        
        # Show result message
        result_text = f"Quiz Finished!\nYour Score: {self.score}/{len(self.questions)}\nLearner Type: {learner_type}"
        messagebox.showinfo("Results", result_text)
        self.root.quit()

    def classify_student(self, accuracy):
        if accuracy >= 0.8:
            return "Advanced Learner - Highly Engaged"
        elif accuracy >= 0.5:
            return "Active Learner - Steady Progress"
        else:
            return "Passive Learner - Needs Improvement"

# Main GUI loop
root = tk.Tk()
app = QuizApp(root)
root.mainloop()
