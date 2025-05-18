import random
import json
import torch 
from model import NeuralNet
from nltk_util import bag_of_words, tokenize, remove_stopwords, lemmatize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sqlite3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents from the JSON file
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Load trained model and data
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

# Initialize the model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Bot name and introductory message
bot_name = "Your Educational counselor"
print("Let's chat! Type 'quit' to exit!")

def get_cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    return cosine_similarity([vec1], [vec2])[0][0]

def store_unknown_message(message):
    """Store unrecognized messages in the database."""
    # Join the list of words into a single string
    sentence_str = ' '.join(message)  # Converts the list into a string
    
    # Connect to the database (or create it)
    conn = sqlite3.connect('unknown_words.db')
    cursor = conn.cursor()

    # Insert the new unknown word into the database
    cursor.execute("INSERT INTO unknown_words (string, val) VALUES (?, ?)", (sentence_str, 0))
    conn.commit()
    conn.close()

while True:
    sentence = input('You: ')
    
    if sentence == "quit":
        break

    # Tokenize the sentence
    sentence = tokenize(sentence)
    
    # Remove stopwords and lemmatize
    sentence = remove_stopwords(sentence)  # Remove stopwords
    sentence = [lemmatize(word) for word in sentence]  # Lemmatize the words

    # Convert the sentence to a bag-of-words format
    input_bag_of_words = bag_of_words(sentence, all_words)

    # Initialize best similarity score and best tag
    best_similarity = 0.0
    best_tag = None

    # Compare input sentence with all known patterns
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            # Convert the pattern to bag of words
            pattern_bag_of_words = bag_of_words(tokenize(pattern), all_words)
            
            # Calculate cosine similarity between the input sentence and the pattern
            similarity = get_cosine_similarity(input_bag_of_words, pattern_bag_of_words)
            
            # Track the best matching intent based on similarity
            if similarity > best_similarity:
                best_similarity = similarity
                best_tag = intent["tag"]

    # Define a threshold for similarity to identify unrecognized input
    similarity_threshold = 0.7  # You can adjust this value based on results

    # If similarity is above threshold, use the best matching intent
    if best_similarity > similarity_threshold:
        # Get response for the best matching intent
        for intent in intents["intents"]:
            if best_tag == intent["tag"]:
                print(f'{bot_name}: {random.choice(intent["responses"])}')
    else:
        # If no good match is found, respond with "I do not understand..."
        print(f"{bot_name}: I do not understand..")
        store_unknown_message(sentence)  # Store unknown message in the database
