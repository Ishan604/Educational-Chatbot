import sqlite3

def create_db():
    # Create the SQLite database
    conn = sqlite3.connect('unknown_words.db')
    cursor = conn.cursor()

    # Create a table to store unknown words
    cursor.execute('''CREATE TABLE IF NOT EXISTS unknown_words (
                        id INTEGER PRIMARY KEY,
                        string TEXT,
                        val INTEGER)''')
    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_db()
