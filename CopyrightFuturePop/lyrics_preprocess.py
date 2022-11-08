import os
import nltk
from nltk.lm.preprocessing import pad_both_ends

DATA_PATH = "pop-lyrics-dataset/lyrics"
MIN_WORDS = 2

def get_all_english_songs_files():
    """Get all songs files from the dataset"""
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith(".txt"):
                yield os.path.join(root, file)

def get_all_sentences():
    """Get all songs lines from the dataset"""
    for file in get_all_english_songs_files():
        with open(file, "r", encoding="utf8") as f:
            if not is_language_english(f.readline()):
                continue
            f.seek(0)
            for line in f.readlines():
                tokens = nltk.word_tokenize(line)
                if len(tokens) > MIN_WORDS:
                    yield pad_both_ends(tokens, 2)

def is_language_english(text):
    """Check heuristicly if the language of the line is english"""
    try:
        text.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True
