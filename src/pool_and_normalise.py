import numpy as np
import torch
from sklearn.preprocessing import normalize

# This file contains the functions for performing pooling and
# normalisation of attribute-dervied text embeddings.
# If only one embedding exists, only normalisation is applied.

# Master function for performing pooling and normalisation.
# Accepts a list of text embeddings and the required operation - 'mean'
# or 'normalise_only'.
def pool_and_normalise(sentence_embeddings, op):
    if op not in ['mean', 'normalise_only']:
        raise ValueError("Invalid argument - 'element_wise_op' must have value of 'mean' or 'normalise_only'.")
    
    # If there are no embeddings in the list raise an error.
    if len(sentence_embeddings) == 0:
        raise ValueError("Input list is empty.")
    
    # If the embedding is a PyTorch tensor object convert it to a NumPy array.
    if torch.is_tensor(sentence_embeddings) == True:
        try:
            vectors = sentence_embeddings.cpu().detach().numpy()
        except Exception as e:
            print(f"""Unable to convert PyTorch tensor to numpy array using 
                    sentence_embeddings.cpu().detach().numpy() - {e}.
                    Trying with sentence_embeddings.detach().numpy()...""")
            try:
                vectors = sentence_embeddings.detach().numpy()
                
            except:
                raise RuntimeError("Unable to convert PyTorch tensor to numpy array.")

    elif torch.is_tensor(sentence_embeddings) == False: 
        vectors = sentence_embeddings
    
    # For preprocessing techniques that compute one single string from all attribute triples
    if op == "normalise_only":
        output_vector = vectors
        return output_vector
    elif op == "mean":
        # If there is only one embedding then normalise it and return it 
        if len(vectors) == 1:
            output_vector = np.array(vectors[0]) 
            output_vector = normalize(output_vector, norm='l2', axis=1)
            return output_vector
        
        # If there's more than one embedding then perform mean pooling and normalise
        # Ensure all vectors have equal dimensionality
        dimension = len(vectors[0])
        if not all(len(v) == dimension for v in vectors):
            raise ValueError("Input vectors are not of equal dimensionality.")
                    
        # Initialize the output vector as a 0 vector of the same length as the passed vector
        output_vector = np.array([0] * len(vectors[0]))
        for vec in vectors[1:]:
            # Sum
            output_vector = [x + y for x, y in zip(output_vector, vec)]
        # Divide    
        output_vector = np.array([x / len(vectors) for x in output_vector])
        # Normalise the vector to have unit length - this is required prior to encryption and dot product similarity
        output_vector = normalize(output_vector, norm='l2', axis=1)
    
    return output_vector