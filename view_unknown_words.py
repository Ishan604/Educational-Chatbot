import sqlite3

def view_unknown_words():
    # Connect to the database
    conn = sqlite3.connect('unknown_words.db')
    cursor = conn.cursor()

    # Query the database to get all rows from the unknown_words table
    cursor.execute("SELECT * FROM unknown_words")
    rows = cursor.fetchall()

    # Print each row
    print("Stored unknown words in the database:")
    for row in rows:
        print(f"ID: {row[0]}, Message: {row[1]}, Value: {row[2]}")

    conn.close()

if __name__ == "__main__":
    view_unknown_words()
