import preprocessing as pp
import embedding as emb
import inference as inf
import encryption as encr
import torch
import argparse
import time

# This file contains the main function for executing the full
# unsupervised, privacy-preserving, entity alignment solution.

def main(dataset, model_name, sentence_type, remove_value_data_type, element_wise_op, evaluation_direction, enc):
    
    # Perform checks on input arguments.
    if dataset not in ["D_W_15K_V1", "D_W_15K_V2"]:
        raise ValueError("""Invalid argument - 'dataset' must have value of
                         'D_W_15K_V1" or 'D_W_15K_V2'.""")

    if model_name not in ['all-mpnet-base-v2','sentence-t5-large']:
        raise ValueError("""Invalid argument - 'model_name' must have value of
                         'all-mpnet-base-v2' or 'sentence-t5-large'.""")

    if sentence_type in ["attr_val", "val"] and element_wise_op == "normalise_only":
        raise ValueError("Unless you are concatenating all triples into one sentence, " +
                         "'element_wise_op' must have a value of 'mean'.")
    
    if element_wise_op not in ['mean', 'normalise_only']:
        raise ValueError("Invalid argument - 'element_wise_op' must have value of 'mean', or 'normalise_only'.")
    
    if evaluation_direction not in ["dbp_to_wkd", "wkd_to_dbp"]:
        raise ValueError("""Invalid argument - 'evaluation_direction' must have value of
                         'dbp_to_wkd" or 'wkd_to_dbp'.""")

    # Set relative dataset path
    if dataset == "D_W_15K_V1":
        dataset_path = r"../data/D_W_15K_V1"
    elif dataset == "D_W_15K_V2":
        dataset_path = r"../data/D_W_15K_V2"
    elif dataset == "Testing":
        dataset_path = r"../data/Testing"
    
    # If concatenating all attribute triples to one string then set to 'normalise_only' for pooling
    if sentence_type in ["attr_val_one_sentence", "val_one_sentence"]:
        element_wise_op = "normalise_only"
        print("'element_wise_op' defaulted to 'normalise_only' as 'sentence_type' = 'one_sentence'")
    
    # Record the start time of script execution.
    start_time = time.time()

    # Perform preprocessing
    dbp_dict, wkd_dict = pp.preprocess(dataset_path=dataset_path,
                                       sentence_type=sentence_type,
                                       remove_value_data_type=remove_value_data_type)
    
    # Perform embedding of Wikidata attribute triples.
    wkd_embeddings = emb.embed(wkd_dict,
                               model_name=model_name,
                               element_wise_op=element_wise_op)

    # Perform embedding of DBpedia attribute triples.
    dbp_embeddings = emb.embed(dbp_dict,
                               model_name=model_name,
                               element_wise_op=element_wise_op)

    # Perform encryption if argument specifies to do so.
    if enc == True:
        context = encr.context()
        wkd_embeddings = encr.encrypt_dictionary_of_embeddings(wkd_embeddings,
                                                              context)
        dbp_embeddings = encr.encrypt_dictionary_of_embeddings(dbp_embeddings,
                                                              context)

    hits_at_1, hits_at_5, hits_at_10, hits_at_50, mrr = inf.evaluate(dbp_embeddings,
                                                                     wkd_embeddings,
                                                                     dataset_path,
                                                                     evaluation_direction=evaluation_direction,
                                                                     enc=enc)

    # Record the end time of the script.
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution_time = {str(elapsed_time)}s.")

    # Clear the CUDA cache to regulate GPU memory utilisation.
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsupervised Entity Alignment.")
    
    # Define command-line arguments
    parser.add_argument("--dataset", type=str, default="D_W_15K_V1", help="Choose a dataset - 'D_W_15K_V1' or 'D_W_15K_V2'.")
    parser.add_argument("--model_name", type=str, default="sentence-t5-large", help="Choose a model - 'sentence-t5-large' or 'all-mpnet-base-v2'.")
    parser.add_argument("--sentence_type", type=str, default="val_one_sentence", help="Choose a preprocessing technique - 'attr_val_one_sentence', 'attr_val, or 'val_one_sentence'.")
    parser.add_argument("--remove_value_data_type", action="store_true", help="Boolean - removes the value type from attribute values.")
    parser.add_argument("--element_wise_op", type=str, default="mean", help="Pooling operation.")
    parser.add_argument("--evaluation_direction", type=str, default="dbp_to_wkd", help="Direction to compare entities. Either 'dbp_to_wkd' or 'wkd_to_dbp'.")
    parser.add_argument("--enc", action="store_true", help="Boolean - run encrypted or unencrypted.")

    args = parser.parse_args()
    for arg_name, arg_value in vars(args).items():
        print(f"{arg_name}: {arg_value}")
        
    # Call your main function with command-line arguments
    main(args.dataset, args.model_name, args.sentence_type, args.remove_value_data_type, args.element_wise_op, args.evaluation_direction, args.enc)

