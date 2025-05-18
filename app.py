from flask import Flask, render_template, request, jsonify
import torch
from model import NeuralNet
from nltk_util import bag_of_words, tokenize, remove_stopwords, lemmatize
import json
import random
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
import subprocess

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Your Educational counselor"

def init_db():
    conn = sqlite3.connect('unknown_words.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS unknown_words (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            string TEXT NOT NULL,
            val INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def store_unknown_message(message):
    sentence_str = ' '.join(message)
    conn = sqlite3.connect('unknown_words.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO unknown_words (string, val) VALUES (?, ?)", (sentence_str, 0))
    conn.commit()
    conn.close()

def get_cosine_similarity(vec1, vec2):
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity([vec1], [vec2])[0][0]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["GET"])
def get_bot_response():
    user_input = request.args.get('msg')
    sentence = tokenize(user_input)
    sentence = remove_stopwords(sentence)
    sentence = [lemmatize(word) for word in sentence]
    input_bag_of_words = bag_of_words(sentence, all_words)
    best_similarity = 0.0
    best_tag = None

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            pattern_bag_of_words = bag_of_words(tokenize(pattern), all_words)
            similarity = get_cosine_similarity(input_bag_of_words, pattern_bag_of_words)
            if similarity > best_similarity:
                best_similarity = similarity
                best_tag = intent["tag"]

    similarity_threshold = 0.7
    if best_similarity > similarity_threshold:
        for intent in intents["intents"]:
            if best_tag == intent["tag"]:
                response = random.choice(intent["responses"])
                return jsonify({"response": response})
    else:
        store_unknown_message(sentence)
        return jsonify({"response": "I do not understand.."})

@app.route('/admin')
def admin_dashboard():
    return render_template('admin_dashboard.html')

@app.route('/get_unknown_questions')
def get_unknown_questions():
    conn = sqlite3.connect('unknown_words.db')
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT string FROM unknown_words WHERE val = 0")
    questions = [row[0] for row in cursor.fetchall()]
    conn.close()
    return jsonify({"questions": questions})

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    data = request.get_json()
    question = data.get('question')
    answer = data.get('answer')

    if not question or not answer:
        return jsonify({'message': 'Missing question or answer.'}), 400

    with open('intents.json', 'r') as f:
        intents_data = json.load(f)

    new_tag = "_".join(question.lower().split())
    new_intent = {
        "tag": new_tag,
        "patterns": [question],
        "responses": [answer]
    }
    intents_data['intents'].append(new_intent)

    with open('intents.json', 'w') as f:
        json.dump(intents_data, f, indent=4)

    conn = sqlite3.connect('unknown_words.db')
    cursor = conn.cursor()
    cursor.execute("UPDATE unknown_words SET val = 1 WHERE string = ?", (question,))
    conn.commit()
    conn.close()

    return jsonify({'message': 'Answer submitted and added to intents.json.'})

@app.route('/retrain_now', methods=['POST'])
def retrain_now():
    try:
        result = subprocess.run(["python", "update_and_retrain.py"], capture_output=True, text=True)
        if result.returncode == 0:
            return jsonify({"message": "Model retrained successfully."})
        else:
            return jsonify({"message": f"Retraining failed: {result.stderr}"}), 500
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
