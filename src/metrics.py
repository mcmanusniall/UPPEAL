import heapq

# This file contains functions for the evaluation metrics.

# Computes the hits@x given a dictionary of the format
# {source entity : {target entity : similarity score}}.
# Returns an int object of the total number of hits.
def hits_at_x(embedding_similarities, linked_entity_pairs_dict, x):
    hits = 0
    for entity, entity_similarities_dict in embedding_similarities.items():
        hits_at_x = heapq.nlargest(x, entity_similarities_dict, key = entity_similarities_dict.get)
        if linked_entity_pairs_dict[entity] in hits_at_x:
            # print("It matched!")
            hits += 1
        else:
            pass
            # print("No match.")
    
    hits_percentage = hits / len(linked_entity_pairs_dict)
    print(f"Total Hits @ {str(x)}: {hits}.")
    print(f"Percentage of Hits @ {str(x)}: {str(hits_percentage)}.")
    
    return hits

# Computes the mean reciprocal rank given a dictionary of the format
# {source entity : {target entity : similarity score}}.
def mean_reciprocal_rank(embedding_similarities, linked_entity_pairs_dict):
    print("Computing MRR...")
    ranks = []
    for entity, entity_similarities_dict in embedding_similarities.items():
        sorted_similarities = sorted(entity_similarities_dict.items(), key=lambda x: x[1], reverse=True)
        linked_entity = linked_entity_pairs_dict[entity]
        rank = 0
        for i, (ent, scores) in enumerate(sorted_similarities, 1):
            if ent == linked_entity:
                rank = i
        
        residual_rank = 1 / rank
        ranks.append(residual_rank)
            
    mrr = sum(ranks) / len(ranks)
    print(f"Mean Reciprocal Rank (MRR) = {round(mrr, 4)}")
    
    return mrr