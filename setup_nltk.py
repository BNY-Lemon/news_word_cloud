import nltk
from nltk.data import find

def ensure_stopwords():
    try:
        # Check if stopwords are already available
        find("corpora/stopwords")
        print("NLTK stopwords already installed.")
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download("stopwords")

def ensure_nltk_resources():
    resources = ["corpora/stopwords", "tokenizers/punkt"]
    for resource in resources:
        try:
            find(resource)
            print(f"NLTK {resource} already installed.")
        except LookupError:
            print(f"Downloading NLTK {resource}...")
            nltk.download(resource.split("/")[1])

if __name__ == "__main__":
    ensure_stopwords()