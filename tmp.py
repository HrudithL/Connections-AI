import requests
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# ConceptNet API base URL
BASE_URL = "http://api.conceptnet.io"
def fetch_conceptnet_data(word):
    """
    Fetch related data for a word from ConceptNet API.
    """
    url = f"{BASE_URL}/c/en/{word.lower()}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['edges']
    else:
        print(f"Failed to fetch data for {word}")
        return []

def extract_features(word):
    """
    Extract semantic relationships and create a feature vector for a word.
    """
    edges = fetch_conceptnet_data(word)
    related_words = set()
    relationships = {
        'RelatedTo': 0,
        'IsA': 0,
        'Synonym': 0,
        'PartOf': 0,
        'HasContext': 0,
        'Antonym': 0,
        'DerivedFrom': 0
    }

    for edge in edges:
        rel = edge['rel']['label']
        end_word = edge['end']['label'].lower()
        start_word = edge['start']['label'].lower()

        # Skip if the end word is the same as the input word
        if end_word == word.lower():
            continue

        if rel in relationships:
            relationships[rel] += 1
            related_words.add(end_word)

    # Create a feature vector (counts of specific relationships)
    feature_vector = np.array(list(relationships.values()))
    return feature_vector, list(related_words)

def build_similarity_matrix(words, word_features):
    """
    Create a similarity matrix for a list of words based on their feature vectors.
    """
    # Extract feature vectors
    feature_vectors = [word_features[word]['features'] for word in words]

    # Normalize feature vectors
    normalized_vectors = normalize(feature_vectors)

    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(normalized_vectors)

    return similarity_matrix

def load_embeddings(file_path):
    """
    Load ConceptNet Numberbatch embeddings from a file into a dictionary.
    """
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 301:  # Skip malformed lines
                continue
            word = parts[0]
            vector = np.array(parts[1:], dtype=float)
            embeddings[word] = vector
    print(f"Loaded {len(embeddings)} word embeddings.")
    return embeddings

def get_embedding(word, embeddings):
    """
    Retrieve the word embedding from the loaded embeddings dictionary.
    """
    return embeddings.get(word.lower(), np.zeros(300))

def create_feature_vector(word, word_features, embeddings):
    """
    Create an enriched feature vector for a word by combining:
    - Semantic relationship counts (from ConceptNet API)
    - Pre-trained embeddings (from ConceptNet Numberbatch file)
    """
    # Semantic relationship features (7-dimensional vector)
    semantic_features = word_features[word]['features']

    # Embedding features (300-dimensional vector)
    embedding_features = get_embedding(word, embeddings)

    # Concatenate semantic features with embedding features
    feature_vector = np.concatenate([semantic_features, embedding_features])

    return feature_vector

def build_feature_matrix(words, word_features, embeddings):
    """
    Construct the final feature matrix for the list of words.
    """
    feature_matrix = np.array([create_feature_vector(word, word_features, embeddings) for word in words])

    # Normalize the feature matrix
    feature_matrix = normalize(feature_matrix)

    return feature_matrix

def compute_similarity_matrix(feature_matrix):
    """
    Compute a pairwise similarity matrix using cosine similarity.
    """
    return cosine_similarity(feature_matrix)





# Example usage:
words = ["apple", "banana", "car", "bus"]
word_features = {
    "apple": {"features": np.array([5, 3, 2, 1, 0, 0, 0])},
    "banana": {"features": np.array([4, 2, 3, 1, 0, 0, 0])},
    "car": {"features": np.array([7, 2, 1, 4, 0, 1, 1])},
    "bus": {"features": np.array([6, 1, 2, 5, 0, 0, 1])}
}

# Build feature matrix and compute similarity matrix
feature_matrix = build_feature_matrix(words, word_features, load_embeddings("numberbatch.txt"))
similarity_matrix = compute_similarity_matrix(feature_matrix)

print("Feature Matrix Shape:", feature_matrix.shape)
print("Similarity Matrix:")
print(np.round(similarity_matrix, 2))