import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors

path_to_embeddings = 'numberbatch.txt'  # Update with the path to your file
conceptnet_model = KeyedVectors.load_word2vec_format(path_to_embeddings, binary=False)

# Example embedding function (ensure ConceptNet embeddings are loaded as shown previously)
def get_embedding(word, model=conceptnet_model, vector_size=300):
    try:
        return model[word]
    except KeyError:
        return np.zeros(vector_size)

def model(words, strikes, isOneAway, correctGroups, previousGuesses, error):
    """
    Model function to generate a guess of 4 words based on word embeddings and similarity.
    
    Parameters:
        words (list): 1D Array with 16 shuffled words.
        strikes (int): Number of strikes.
        isOneAway (bool): Indicates if the previous guess was close.
        correctGroups (list): 2D Array of previously guessed correct groups.
        previousGuesses (list): 2D Array of previous guesses.
        error (str): Error message (0 if no error).
    
    Returns:
        guess (list): 1D Array of 4 words.
        endTurn (bool): Boolean indicating if the turn should end.
    """
    
    # Gets vectorization for each word and puts it in a dictionary. 
    word_vectors = {word: get_embedding(word) for word in words}

    # Calculates similarity of words
    word_list = list(word_vectors.keys())
    word_matrix = np.vstack(list(word_vectors.values()))  # Create matrix of vectors
    
    # Compute pairwise cosine similarity <-- yeah wtvr tf that means
    similarity_matrix = cosine_similarity(word_matrix)
    
    # Highest similarity words from the given
    max_similarity = 0
    best_guess = []
    
    # Iterate over all possible combinations of 4 words <-- this may or may not be stupid
    for i in range(len(word_list)):
        for j in range(i + 1, len(word_list)):
            for k in range(j + 1, len(word_list)):
                for l in range(k + 1, len(word_list)):
                    # Sum similarity of the selected 4-word group
                    selected_indices = [i, j, k, l]
                    selected_words = [word_list[idx] for idx in selected_indices]
                    
                    # Calculate total similarity score for the group
                    similarity_score = (similarity_matrix[i, j] + similarity_matrix[i, k] +
                                        similarity_matrix[i, l] + similarity_matrix[j, k] +
                                        similarity_matrix[j, l] + similarity_matrix[k, l])
                    
                    # Check if this is the highest similarity group and avoid previous guesses
                    if similarity_score > max_similarity and selected_words not in previousGuesses:
                        max_similarity = similarity_score
                        best_guess = selected_words

    # Set the best guess
    guess = best_guess if best_guess else words[:4]  # Fallback to first 4 words if no best guess found
    endTurn = False  # Set endTurn to False to allow further attempts if its that bad
    
    # Check for conditions based on strikes, error, or isOneAway
    if strikes >= 3 or error != "0":
        endTurn = True  # End the turn if maximum strikes reached or there is an error
    
    return guess, endTurn
