from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def tokenize(text):
    stop_words = set(stopwords.words('english'))
    tokenized = [token.lower() for token in word_tokenize(text) if token.isalpha()]
    filtered = [token for token in tokenized if not token in stop_words]
    return filtered