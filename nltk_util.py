import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import numpy as np

# nltk.download('punkt')
# nltk.download('punkt_tab') 
# nltk.download('stopwords')
# nltk.download('wordnet')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Assign unique values stopwords in English to a variable for later use
stop_words = set(stopwords.words('english'))

def tokenize(sentence):
    """
    Tokenize the input sentence into words.

    Returns:
    list: A list of tokenized words.
    """
    return nltk.word_tokenize(sentence)  

def stem(words):
    """
    Stem the input words to their root form using the PorterStemmer.

    Returns:
    str: The stemmed word.
    """
    return stemmer.stem(words.lower())

def lemmatize(words):
    """
    Lemmatize the input words to their base form using the WordNetLemmatizer.

    Returns:
    str: The lemmatized word.
    """
    return lemmatizer.lemmatize(words.lower())

# Function to remove stopwords from a sentence
def remove_stopwords(tokens):
    """
    Remove stopwords from a list of tokens (words).

    Returns:
    list: A list of words with stopwords removed.
    """
    return [word for word in tokens if word not in stop_words]

def bag_of_words(tokenized_sentence, all_words):
    """
    Convert a tokenized sentence into a bag of words representation.
    This method will check which words from the sentence exist in the 'all_words' list,
    and return a binary representation of the sentence.
    Args:
    tokenized_sentence (list): A list of words from a tokenized sentence.
    all_words (list): A list of all possible words to compare the sentence's words against.
    Returns:
    numpy.ndarray: An array with binary values representing the presence of words from 'all_words' in the sentence.
    """
    # Stemming each word in the tokenized sentence
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    # Remove stopwords from the tokenized sentence
    tokenized_sentence = remove_stopwords(tokenized_sentence)

    # Create a bag of words with 0s initially
    bag = np.zeros(len(all_words), dtype=np.float32)

    # Loop through all_words and check if each word is present in the tokenized_sentence
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0  # Set 1 if word is found

    return bag


# sentence = ["hello", "how", "are", "you"]
# words = ["Hi", "hello", "I", "you", "bye", "thank", "cool"]
# bag = bag_of_words(sentence, words)
# print("Bag of words: ", bag)
