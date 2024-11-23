import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding
from tensorflow.keras.optimizers import Adam
import tkinter as tk
from tkinter import messagebox


num_questions = 10  
embedding_dim = 32  
rnn_units = 64 


X_train = np.random.randint(0, num_questions, (500, 10)) 
y_train = np.random.randint(0, 2, (500, 10))
model = Sequential([
    Embedding(input_dim=num_questions, output_dim=embedding_dim, input_length=10),  # Question Embedding
    GRU(rnn_units, return_sequences=False),  # GRU layer to track knowledge state
    Dense(num_questions, activation='sigmoid')  
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)  # Training the model on dummy data


class QuizApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Learning Capability Quiz")
        
        
        self.question_index = 0
        self.score = 0
        self.question_ids = []
        self.correct_answers = []
        self.knowledge_state = np.zeros((1, rnn_units))  # Initial knowledge state (hidden state of GRU)

        
        self.questions = [
            {"id": 0, "text": "What particles are found in the nucleus?", "answer": "Protons and Neutrons", "options": ["Electrons and Protons", "Protons and Neutrons", "Neutrons and Electrons", "Protons only"], "difficulty": 0.3},
            {"id": 1, "text": "Which force binds protons and neutrons?", "answer": "Nuclear force", "options": ["Gravitational force", "Electromagnetic force", "Nuclear force", "Weak force"], "difficulty": 0.7},
            {"id": 2, "text": "What is the process of splitting a heavy nucleus?", "answer": "Nuclear fission", "options": ["Nuclear fusion", "Nuclear fission", "Radioactive decay", "Beta decay"], "difficulty": 0.8},
            {"id": 3, "text": "What type of radiation consists of helium nuclei?", "answer": "Alpha radiation", "options": ["Beta radiation", "Gamma radiation", "Alpha radiation", "Neutron radiation"], "difficulty": 0.4},
            {"id": 4, "text": "Which is true for radioactive decay?", "answer": "It is a random and spontaneous process", "options": ["It is influenced by temperature", "It is a random and spontaneous process", "It can be controlled", "It is caused by chemical reactions"], "difficulty": 0.6},
            {"id": 5, "text": "Mass defect is related to?", "answer": "Binding energy", "options": ["Charge", "Density", "Binding energy", "Half-life"], "difficulty": 0.5},
            {"id": 6, "text": "The time for half decay is?", "answer": "Half-life", "options": ["Decay constant", "Lifetime", "Half-life", "Average life"], "difficulty": 0.7},
            {"id": 7, "text": "Fusion reactions occur at?", "answer": "High temperature and high pressure", "options": ["Low temperature", "High temperature", "Low pressure", "High pressure"], "difficulty": 0.9},
            {"id": 8, "text": "Beta decay results in?", "answer": "Electrons or positrons", "options": ["Alpha particles", "Gamma rays", "Electrons or positrons", "Neutrons"], "difficulty": 0.6},
            {"id": 9, "text": "Control rods in nuclear reactors?", "answer": "Absorb neutrons", "options": ["Increase rate", "Absorb neutrons", "Reflect neutrons", "Increase temperature"], "difficulty": 0.8}
        ]
        
        
        self.question_label = tk.Label(root, text="", font=("Arial", 14))
        self.question_label.pack(pady=20)

    
        self.options = tk.StringVar()
        self.option_buttons = []
        for i in range(4):
            btn = tk.Radiobutton(root, text="", variable=self.options, value=i, font=("Arial", 12))
            btn.pack(anchor="w")
            self.option_buttons.append(btn)

        
        self.submit_button = tk.Button(root, text="Submit Answer", command=self.submit_answer)
        self.submit_button.pack(pady=20)

        
        self.next_question()

    def next_question(self):
        if self.question_index < len(self.questions):
            
            question = self.questions[self.question_index]
            self.question_label.config(text=question["text"])
            self.question_ids.append(question["id"])

            
            for i, option in enumerate(question["options"]):
                self.option_buttons[i].config(text=option, value=option)
            
            self.options.set("")  # Clear previous selection
        else:
    
            self.display_results()

    def submit_answer(self):
        selected_option = self.options.get()
        if selected_option:
            question = self.questions[self.question_index]
            correct = selected_option == question["answer"]

            
            self.correct_answers.append(int(correct))
            if correct:
                self.score += 1

           
            answer_input = np.array([self.question_ids])
            knowledge_state_prediction = model.predict(answer_input)  # Predict knowledge state

            self.knowledge_state = knowledge_state_prediction

            
            self.question_index += 1
            self.next_question()
        else:
            messagebox.showwarning("Warning", "Please select an answer.")

    def display_results(self):
        
        avg_correctness = np.mean(self.correct_answers)

        
        learner_type = self.classify_student(avg_correctness)
        
    
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


root = tk.Tk()
app = QuizApp(root)
root.mainloop()
