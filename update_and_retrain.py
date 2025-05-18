import sqlite3
import json
import os

def update_intents_with_unhandled_words(): 
    """Fetch unhandled unknown questions and append them with default response."""
    conn = sqlite3.connect('unknown_words.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM unknown_words WHERE val = 0")
    unknown_entries = cursor.fetchall()
    conn.close()

    if not unknown_entries:
        print("No new unknown questions found.")
        return

    with open('intents.json', 'r') as f:
        intents = json.load(f)

    for entry in unknown_entries:
        question = entry[1]
        tag = "auto_unknown_" + "_".join(question.lower().split())

        # Only add if not already present
        if not any(tag == intent["tag"] for intent in intents["intents"]):
            intents["intents"].append({
                "tag": tag,
                "patterns": [question],
                "responses": ["I do not understand..."]
            })

    with open('intents.json', 'w') as f:
        json.dump(intents, f, indent=4)

    print(f"Added {len(unknown_entries)} unknown questions.")

def retrain_model():
    print("Retraining model...")
    os.system('python train.py')

def update_and_retrain():
    update_intents_with_unhandled_words()
    retrain_model()

if __name__ == "__main__":
    update_and_retrain()

