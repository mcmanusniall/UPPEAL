import torch
import os
import numpy as np
import pool_and_normalise as pool
from sentence_transformers import SentenceTransformer

# This file manages the embedding of attribute triples using pre-trained language models.
# Two pre-trained language models are available from the sentence-transfomers library:
# AllMPNet-Base-v2 (https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
# and Sentence-T5-Large (https://huggingface.co/sentence-transformers/sentence-t5-large/tree/main).
# If an NVIDIA GPU is available, it will be utilised for embedding.

# Master function for generating embeddings.
def embed(dictionary, model_name, element_wise_op):
    # Helps regulate GPU memory allocation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    if model_name == "all-mpnet-base-v2":
        model_name == "sentence-transformers/all-mpnet-base-v2"
    elif model_name == "sentence-t5-large":
        model_name == "sentence-transformers/sentence-t5-large"

    embeddings = generate_embeddings(dictionary, element_wise_op, model_name)
    
    return embeddings

# Generates embeddings given a dictionary of the format {entity : [attribute triples]}.
# Set element_wise_op to 'mean' to perform mean pooling if there are multiple
# attribute triples, or 'normalise_only' if all attributes have been concatenated
# into one string.
def generate_embeddings(dictionary, element_wise_op, model_name):
    # Check to see if an NVIDIA GPU is available.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cuda:0":
        print("Utilising GPU for tokenization and embedding.")
    elif device == "cpu":
        print("No GPU available - utilising CPU for tokenization and embedding.")
    
    # Get the model configuration
    model = SentenceTransformer(model_name)
    print(f"Generating embeddings with {str(model_name)}...")
    embedding_dictionary = {}
    for entity, sentences in dictionary.items():
        # Generates embeddings
        embedded_sentences = model.encode(sentences,
                                          batch_size=32,
                                          show_progress_bar=None,
                                          output_value='sentence_embedding',
                                          convert_to_numpy=False,
                                          convert_to_tensor=True, 
                                          device=device,
                                          normalize_embeddings=False)
        
        # Performs mean pooling and normalisation of embeddings
        final_embedding = pool.pool_and_normalise(embedded_sentences,
                                                  element_wise_op)
        
        # Add the sentence embedding to the new dictionary    
        embedding_dictionary[str(entity)] = final_embedding
        
        try:
            # Clears the CUDA cache if a GPU is being used - this
            # helps regulate GPU memory
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"An error occurred when trying to clear the CUDA cache: {e}.")
        
    print("Successful!")
        
    return embedding_dictionary