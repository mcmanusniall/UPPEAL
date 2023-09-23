import numpy as np
import preprocessing as pp
import metrics as metrics

# This file contains functions to perform inference and evaluation of the solution.

# Computes the dot product similarity for UNENCRYPTED embeddings.
# Accepts two dictionaries of the format {entity : unencrypted embedding}.
# compare_these_entities is the source KG.
# with_these_entities is the target KG.
# Returns a dictionary of the format {source_entity : {target_entity : similarity}}
def dot_product_of_summed_embeddings(compare_these_entities, with_these_entities):
    print("Computing dot product similarities between embeddings...")
    
    # Extract the embeddings from the two dictionaries and converts
    # to a numpy array.
    compare_embeddings = np.array(list(compare_these_entities.values()))
    with_embeddings = np.array(list(with_these_entities.values()))
    
    # Compute the dot product similarity
    similarities = np.dot(compare_embeddings, with_embeddings.T)
    
    embedding_similarities = {}
    for i, entity in enumerate(compare_these_entities):
        similarities_for_entity = {}
        for j, entity_to_compare in enumerate(with_these_entities):
            similarity = similarities[i, j]
            similarities_for_entity[entity_to_compare] = similarity
        
        embedding_similarities[entity] = similarities_for_entity
    print("Completed.")
    return embedding_similarities

# Computes the dot product similarity between ENCRYPTED embeddings.
# This method is computationally feasible but is not replicative of how
# the solution can be applied in a real-world application as it performs
# decryption prior to computing the similarity.
# This method still replicates the approximation that is introduced when encyrpting embeddings.
# Returns a dictionary of the format {source_entity : {target_entity : similarity}}
def dot_product_of_encrypted_embeddings(compare_these_entities, with_these_entities):
    print("Computing dot product similarity between encrypted embeddings...")

    embedding_similarities = {}
    for entity, enc_embedding in compare_these_entities.items():
        similarities_for_entity = {}
        enc_embedding = enc_embedding.decrypt()
        for entity_to_compare, enc_embedding_to_compare in with_these_entities.items():
            similarity = np.dot(enc_embedding, enc_embedding_to_compare.decrypt())
            similarities_for_entity[entity_to_compare] = similarity

        embedding_similarities[entity] = similarities_for_entity

    print("Completed.")

    return embedding_similarities


# Computes the dot product similarity between ENCRYPTED embeddings.
# This method is computationally infeasible but IS replicative of how
# the solution can be applied in a real-world application.
# Returns a dictionary of the format {source_entity : {target_entity : similarity}}
#
# # Computes the dot product similarity for ENCRYPTED embeddings
# def dot_product_of_encrypted_embeddings(compare_these_entities, with_these_entities):
#     print("Computing dot product similarity between encrypted embeddings...")

#     embedding_similarities = {}
#     for entity, enc_embedding in compare_these_entities.items():
#         similarities_for_entity = {}
#         for entity_to_compare, enc_embedding_to_compare in with_these_entities.items():
#             enc_similarity = enc_embedding.dot(enc_embedding_to_compare)
#             similarity = enc_similarity.decrypt()[0]
#             similarities_for_entity[entity_to_compare] = similarity

#         embedding_similarities[entity] = similarities_for_entity

#     print("Completed.")

#     return embedding_similarities

# Master function that calculates the similarity between two dictionaries
# of the format {entity : embedding}.
# Evaluates the performance of the solution using the metrics hits@1, hits@5
# hits@10, hits@50, and MRR and returns these metrics.
# enc is a boolean that specifies if the embeddings are encrypted or not.
def evaluate(dbp_embeddings, wkd_embeddings, dataset_path, evaluation_direction, enc):
    if evaluation_direction not in ['dbp_to_wkd', 'wkd_to_dbp']:
        raise ValueError("Invalid argument - 'evaluation_direction' must have value of 'dbp_to_wkd' or 'wkd_to_dbp'.")
    
    if evaluation_direction == "dbp_to_wkd":
        if enc == False:
            embedding_similarities = dot_product_of_summed_embeddings(dbp_embeddings,
                                                                      wkd_embeddings)
        elif enc == True:
            embedding_similarities = dot_product_of_encrypted_embeddings(dbp_embeddings,
                                                                         wkd_embeddings)
    elif evaluation_direction == "wkd_to_dbp":
        if enc == False:
            embedding_similarities = dot_product_of_summed_embeddings(wkd_embeddings,
                                                                      dbp_embeddings)
        elif enc == True:
            embedding_similarities = dot_product_of_encrypted_embeddings(wkd_embeddings,
                                                                         dbp_embeddings)
    
    # Compute dictionary of entity links based on the provided mapping
    entity_links_clean = dataset_path + "/ent_links_clean"
    linked_ents_dict = pp.get_linked_entity_pairs_as_dict(entity_links_clean,
                                                          evaluation_direction)
    
    hits_at_1 = metrics.hits_at_x(embedding_similarities, linked_ents_dict, 1)
    hits_at_5 = metrics.hits_at_x(embedding_similarities, linked_ents_dict, 5)
    hits_at_10 = metrics.hits_at_x(embedding_similarities, linked_ents_dict, 10)
    hits_at_50 = metrics.hits_at_x(embedding_similarities, linked_ents_dict, 50)
    mrr = metrics.mean_reciprocal_rank(embedding_similarities, linked_ents_dict)
    
    
    return hits_at_1, hits_at_5, hits_at_10, hits_at_50, mrr

