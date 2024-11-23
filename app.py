from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# In-memory storage for mood responses (for demo purposes)
mood_responses = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit_mood():
    mood = request.form['mood']
    mood_responses.append(mood)  # Store the mood response
    return redirect(url_for('thank_you'))

@app.route('/thank_you')
def thank_you():
    return "<h1>Thank you for your feedback!</h1><a href='/'>Go back</a>"

if __name__ == '__main__':
    app.run(debug=True)
