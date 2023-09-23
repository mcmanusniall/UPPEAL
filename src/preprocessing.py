import os
import logging
import re
import utils as utils
import analysis as analysis
import pickle

# This file contains the functions for preprocessing the DBP-WD-15K

# Reads a file of RDF triples and returns a list of tuples [subject, predicate, object]
def get_triples(filepath):
    try:
        with open(filepath, 'r') as file:
            triples = []
            line_no = 1
            lines = file.readlines()

            for line in lines:
                try:
                    subject, predicate, object = line.strip().split(maxsplit=2)
                    triples.append([subject, predicate, object])
                except Exception as e:
                    logging.exception(f"Error on line {line_no} - '{line}': {e}")
                
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' could not be found.")
        return None
    except Exception as e:
        print(f"Error: An error occurred while reading the file: {e}")
        return None
    
    return triples

# Reads a file containing linked entities and returns a list of tuples [entity_type1, entity_type2]
def get_linked_entity_pairs(filepath):
    try:
        with open(filepath, 'r') as file:
            linked_entity_pairs = []
            line_no = 1
            lines = file.readlines()
            for line in lines:
                try:
                    type_1_entity, type_2_entity = line.strip().split(maxsplit=1)
                    linked_entity_pairs.append([type_1_entity, type_2_entity])
                except Exception as e:
                    logging.exception(f"Error on line {line_no} - '{line}': {e}")
                
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' could not be found.")
        return None
    except Exception as e:
        print(f"Error: An error occurred while reading the file: {e}")
        return None
    
    return linked_entity_pairs

# Returns a dictionary of format {entity_type_1 : linked_entity_type_2} from the file
# containing the linked entity pairs.
# The mapping argument is used to specify which dataset is used as the key and which dataset is used
# as the value. This argument takes values of either 'dbp_to_wkd' or 'wkd_to_dbp'.
def get_linked_entity_pairs_as_dict(filepath, mapping):
    if mapping not in ['dbp_to_wkd', 'wkd_to_dbp']:
        raise ValueError("Invalid string passsed to argument 'mapping' - must be 'dbp_to_wkd' or 'wkd_to_dbp'")
    try:
        with open(filepath, 'r') as file:
            linked_entity_pairs = {}
            line_no = 1
            lines = file.readlines()
            for line in lines:
                try:
                    type_1_entity, type_2_entity = line.strip().split(maxsplit=1)
                    if mapping == "dbp_to_wkd":
                        linked_entity_pairs[type_1_entity] = type_2_entity
                    elif mapping == "wkd_to_dbp":
                        linked_entity_pairs[type_2_entity] = type_1_entity
                except Exception as e:
                    logging.exception(f"Error on line {line_no} - '{line}': {e}")
                
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' could not be found.")
        return None
    except Exception as e:
        print(f"Error: An error occurred while reading the file: {e}")
        return None
    
    return linked_entity_pairs

# Reads a file containg RDF triples and returns three lists - [subjects] [predicates] [objects]
def get_split_triples(filename):
    try:
        with open(filename, 'r') as file:
            subjects = []
            predicates = []
            objects = []
            lines = file.readlines()
            line_no = 1
            try:
                for line in lines:
                    subject, predicate, objekt = line.strip().split(maxsplit=2)
                    subjects.append(subject)
                    predicates.append(predicate)
                    objects.append(objekt)
                    line_no += 1
                    
            except Exception as e:
                  print(f"Error on line {line_no} - '{line}': {e}")

    except FileNotFoundError:
        print(f"Error: The file '{filename}' could not be found.")
        return None
    except Exception as e:
        print(f"Error: An error occurred while reading the file: {e}")
        return None
    
    return subjects, predicates, objects

# Reads a list containing tuples of linked entities [entity_type1, entity_type2] and returns
# two lists - one containing all entities of the first type and one for all entities of the second type
def split_linked_entity_pairs(linked_entity_pairs):
    type_1_entities = []
    type_2_entities = []
    try:
        for pair in linked_entity_pairs:
            type_1_entities.append(pair[0])
            type_2_entities.append(pair[1])
    
    except Exception as e:
        print(f"Error: An error occurred - {e}")
        
    return type_1_entities, type_2_entities

# Takes a list of entities extracted from the attribute triples of one knowledge graph and a
# list of entities extracted from the linked entity pairs for the same knowledge graph and 
# returns entities that have a link but do not have an attribute triple.
def get_entities_with_no_attribute(entities_from_attribute_triples, entities_with_link):
    print("Finding entities that have a link but no attribute triple...")
    try:
        entities_with_attribute = utils.get_unique_values(entities_from_attribute_triples)
        entities_with_link = utils.get_unique_values(entities_with_link)
        entities_with_link_and_no_attribute = entities_with_link - entities_with_attribute
        print(f"Found {len(entities_with_link_and_no_attribute)} entities that do not have an attribute triple.")

    except Exception as e:
        print(f"Error: An error occurred - {e}")

    return entities_with_link_and_no_attribute

# Gets the linked entity pairs where at least one of the entities doesn't have an attribute triple.
def get_linked_entity_pairs_with_no_attribute(type_1_entities_with_no_attribute, type_2_entities_with_no_attribute, linked_entity_pairs):
    linked_entities_without_attribute = []
    print("Finding linked entity pairs where at least one entity does not have an attribute triple...")
    for pair in linked_entity_pairs:
        if pair[0] in type_1_entities_with_no_attribute or pair[1] in type_2_entities_with_no_attribute:
            linked_entities_without_attribute.append(pair)
    print(f"Found {len(linked_entities_without_attribute)}.")

    return linked_entities_without_attribute

# Gets the linked entity pairs where both entities have at least one attribute triple.
def get_linked_entity_pairs_with_attribute(type_1_entities_with_no_attribute, type_2_entities_with_no_attribute, linked_entity_pairs):
    linked_entities_with_attribute = []
    print("Finding linked entity pairs where both entities have at least one attribute triple...")
    for pair in linked_entity_pairs:
        if pair[0] not in type_1_entities_with_no_attribute and pair[1] not in type_2_entities_with_no_attribute:
            linked_entities_with_attribute.append(pair)
    print(f"Found {len(linked_entities_with_attribute)}.")

    return linked_entities_with_attribute

# Gets the linked entity pairs where both entities have no attribute triple.
def get_isolated_linked_entity_pairs(type_1_entities_with_no_attribute, type_2_entities_with_no_attribute, linked_entity_pairs):
    isolated_linked_entity_pairs = []
    print("Finding linked entity pairs where both entities do not have an attribute triple...")
    for pair in linked_entity_pairs:
        if pair[0] in type_1_entities_with_no_attribute and pair[1] in type_2_entities_with_no_attribute:
            isolated_linked_entity_pairs.append(pair)
    print(f"Found {len(isolated_linked_entity_pairs)}.")

    return isolated_linked_entity_pairs

# Creates a new file containing only linked entity pairs where both entities have an attribute triple.
def create_linked_entities_with_attribute_file(linked_entities_with_attribute, dataset_path):
    output_file = dataset_path + "/ent_links_clean"
    print(f"Creating new linked entiy file at {output_file}...")
    try:
        with open(output_file, 'w') as output_f:
            for entity_pair in linked_entities_with_attribute:
                output_f.write(entity_pair[0] + ' ' + entity_pair[1] + '\n')
        print(f"Data has been written to '{output_file}' successfully.")
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")

# Creates a dictionary where each linked entity is the key and the value is a list of attribute
# triples associated with that entity.
def map_attribute_triples_to_entities(entities, subjects, predicates, objects):
    print("Creating dictionary of entities and their associated attribute triples...")
    entities_and_attribute_triples = {}
    try:
        for entity in entities:
            attribute_triples = []
            for subj, pred, obj in zip(subjects, predicates, objects):
                if str(subj) == str(entity):
                    attribute_triples.append([subj, pred, obj])
                    
            entities_and_attribute_triples[entity] = attribute_triples
    
        print("Successful!")
        
    except Exception as e:
        print(f"An error occurred when mapping attributre triples to entities: {e}") 

    return entities_and_attribute_triples

# Converts the Wikidata attributes from URI's with Property IDs to titles
def wkd_attribute_id_to_label(attribute_id_uri, id_to_label_map):
    attribute_label = id_to_label_map[attribute_id_uri]

    return attribute_label

# Converts a Wikidata triple into a natural language sentence containing only
# the attribute type and the attribute value.
def wkd_lexicalise_attribute_value(attribute, value):
    sentence = str(attribute) + " " + str(value)
        
    return str(sentence)

# Converts a Wikidata triple into a natural language sentence containing only
# the attribute value.
def wkd_lexicalise_value(value):
    sentence = str(value)
        
    return str(sentence)

# Wikidata - Converts a dictionary of format {entity URI : [[entity URI, attribute, value]]} into
# a dictionary of format {entity URI : [sentence, sentence, ...]}
def wkd_convert_triples_to_sentencies(dictionary, id_to_label_map, sentence_type, remove_value_data_type):
    if sentence_type not in ['val', 'val_one_sentence', 'attr_val', 'attr_val_one_sentence']:
        raise ValueError("Invalid argument passed to 'sentence_type' - must be 'val', 'val_one_sentence', 'attr_val', 'attr_val_one_sentence'.")
        
    print("Converting attribute triples to sentences...")
    converted_dict = {}
    try:
        for key, triples in dictionary.items():
            sentences = []
            for index, triple in enumerate(triples, 1):
                entity_resource = utils.get_resource_from_uri(triple[0])
                converted_attribute = wkd_attribute_id_to_label(triple[1], id_to_label_map)
                
                # Removes the data type of the value.
                if remove_value_data_type == True:
                    pattern = r'\^\^<[^>]+>'
                    value = re.sub(pattern, '', str(triple[2]))
                elif remove_value_data_type == False:
                    value = triple[2]
                    
                if sentence_type in ["attr_val", "attr_val_one_sentence"]:
                    sentence = wkd_lexicalise_attribute_value(converted_attribute, value)
                elif sentence_type in ["val", "val_one_sentence"]:
                    sentence = wkd_lexicalise_value(value)
                    
                sentences.append(sentence)
                
            if sentence_type == "attr_val":
                converted_dict[key] = sentences
            elif sentence_type == "attr_val_one_sentence":
                one_sentence = ' '.join(sentences)
                converted_dict[key] = one_sentence
            elif sentence_type == "val":
                converted_dict[key] = sentences
            elif sentence_type == "val_one_sentence":
                one_sentence = ' '.join(sentences)
                converted_dict[key] = one_sentence
            
        print("Successful!")
        
    except Exception as e:
        print(f"An error occurred when converting attribute triples to sentences: {e}") 

    return converted_dict

# Converts a DBpedia triple into a natural language sentence containing only
# the attribute type and value.
def dbp_lexicalise_attribute_value(attribute, value):
    sentence = str(utils.camel_case_to_sentence(attribute) + " " + str(value))
    
    return str(sentence)

# Converts a DBpedia triple into a natural language sentence containing only
# the attribute value.
def dbp_lexicalise_value(value):
    sentence = str(value)
    
    return str(sentence)

# DBpedia - Converts a dictionary of format {entity URI : [[entity URI, attribute, value]]} into
# a dictionary of format {entity URI : [sentence, sentence, ...]}
def dbp_convert_triples_to_sentencies(dictionary, sentence_type, remove_value_data_type):
    
    if sentence_type not in ['val', 'val_one_sentence', 'attr_val', 'attr_val_one_sentence']:
        raise ValueError("Invalid argument passed to 'sentence_type' - must be 'val', 'val_one_sentence', 'attr_val', 'attr_val_one_sentence'.")
        
    print("Converting attribute triples to sentences...")
    converted_dict = {}
    try:
        for key, triples in dictionary.items():
            sentences = []
            for index, triple in enumerate(triples, 1):
                entity_resource = utils.get_resource_from_uri(triple[0])
                attribute_resource = utils.get_resource_from_uri(triple[1])
                converted_attribute = utils.camel_case_to_sentence(attribute_resource)
                
                # Removes the data type of the value.
                if remove_value_data_type == True:
                    pattern = r'\^\^<[^>]+>'
                    value = re.sub(pattern, '', str(triple[2]))
                elif remove_value_data_type == False:
                    value = triple[2]
                
                if sentence_type in ["attr_val", "attr_val_one_sentence"]:
                    sentence = dbp_lexicalise_attribute_value(converted_attribute, value)
                elif sentence_type in ["val", "val_one_sentence"]:
                    sentence = dbp_lexicalise_value(value)
                    
                sentences.append(sentence)
            
            if sentence_type == "attr_val":
                converted_dict[key] = sentences
            elif sentence_type == "attr_val_one_sentence":
                one_sentence = ' '.join(sentences)
                converted_dict[key] = one_sentence
            elif sentence_type == "val":
                converted_dict[key] = sentences
            elif sentence_type == "val_one_sentence":
                one_sentence = ' '.join(sentences)
                converted_dict[key] = one_sentence
                
        print("Successful!")
        
    except Exception as e:
        print(f"An error occurred when converting attribute triples to sentences: {e}")

    return converted_dict

# Creates a clean attribute triple file that has the null values removed.
def create_clean_triple_file(input_file):
    print("Checking input file for null values...")
    checklist = analysis.check_null_values(input_file)
    
    if len(checklist) == None:
        print("No null values found. Input file is already clean.")
    else:
        lines_to_remove = [i[0] for i in checklist]
        
        output_path = input_file + "_clean"
        utils.remove_lines_from_file(input_file, lines_to_remove, output_path)
        print(f"Clean file created at: {output_path}")
        
# Master function for performing preprocessing.
def preprocess(dataset_path, sentence_type, remove_value_data_type):

    if remove_value_data_type == True:
        val_removed = "val_type_removed"
    elif remove_value_data_type == False:
        val_removed = "val_type"

    # Set the paths of the pickle files.
    dbp_dict_pkl_path = dataset_path + "/dbp_dict_" + sentence_type + "_" + val_removed + ".pkl"
    wkd_dict_pkl_path = dataset_path + "/wkd_dict_" + sentence_type + "_" + val_removed + ".pkl"

    # If preprocessing has already been conducted on the dataset there will be
    # pickle files for the returned {entity : [attributes]} dictionaries for
    # DBpedia and Wikidata - this avoids running preprocessing again.
    if os.path.exists(dbp_dict_pkl_path) & os.path.exists(wkd_dict_pkl_path):
        try:
            with open(dbp_dict_pkl_path, 'rb') as dbp_dict_file:
                dbp_dict = pickle.load(dbp_dict_file)
            with open(wkd_dict_pkl_path, 'rb') as wkd_dict_file:
                wkd_dict = pickle.load(wkd_dict_file)
                print("Got the pickle files of the pre-processed entity:attributes dictionaries for DBpedia and Wikidata.")
        except:
            print("Couldn't load pickle files - continuing with preprocessing.")

        return dbp_dict, wkd_dict

    # Set paths to files.
    dbp_attr_triples_path = dataset_path + "/attr_triples_1"
    wkd_attr_triples_path = dataset_path + "/attr_triples_2"
    linked_entity_pairs_path = dataset_path + "/ent_links"
    
    # Create clean attribute file
    create_clean_triple_file(dbp_attr_triples_path)
    create_clean_triple_file(wkd_attr_triples_path)
    
    # Reset path to clean files
    dbp_attr_triples_path = dataset_path + "/attr_triples_1_clean"
    wkd_attr_triples_path = dataset_path + "/attr_triples_2_clean"
    
    # Get split triples for both datasets
    dbp_ents, dbp_attr, dbp_vals = get_split_triples(dbp_attr_triples_path)
    wkd_ents, wkd_attr, wkd_vals = get_split_triples(wkd_attr_triples_path)
    
    print(f"Number of DBpedia attribute triples: {len(dbp_ents)}.")
    print(f"Number of Wikidata attribute triples: {len(wkd_ents)}.")
    
    # Get the linked entities as a list of pairs.
    linked_ents = get_linked_entity_pairs(linked_entity_pairs_path)
    
    # Split the list of pairs into two lists, one containing the linked entites for DBpedia
    # and one containing the linked entities for Wikidata
    dbp_linked_ents, wkd_linked_ents = split_linked_entity_pairs(linked_ents)
    
    print(f"Number of DBpedia linked entities: {len(dbp_linked_ents)}.")
    print(f"Number of Wikidata linked entities: {len(wkd_linked_ents)}.")
    
    # DBpedia
    # Get the linked entities that do not have an attribute triple by:
    # 1. Getting the unique values of entities in the split attribute triples list.
    dbp_ents_with_attr = utils.get_unique_values(dbp_ents)
    # 2. Getting the unique values of entities in the split linked entities list.
    dbp_linked_ents_set = utils.get_unique_values(dbp_linked_ents)
    # 3. Calculating the difference i.e. returning those entities that have a link but no attribute.
    dbp_linked_ents_no_attr = dbp_linked_ents_set - dbp_ents_with_attr

    print(f"The number of DBpedia entities that have an attribute triple: {len(dbp_ents_with_attr)}.")
    print(f"The number of DBpedia entities that have a linked entity in Wikidata: {len(dbp_linked_ents_set)}.")
    print(f"The number of DBpedia entities that do not have an attribute triple: {len(dbp_linked_ents_no_attr)}.")
    
    # Wikidata
    # Get the linked entities that do not have an attribute triple by:
    # 1. Getting the unique values of entities in the split attribute triples list.
    wkd_ents_with_attr = utils.get_unique_values(wkd_ents)
    # 2. Getting the unique values of entities in the split linked entities list.
    wkd_linked_ents_set = utils.get_unique_values(wkd_linked_ents)
    # 3. Calculating the difference i.e. returning those entities that have a link but no attribute.
    wkd_linked_ents_no_attr = wkd_linked_ents_set - wkd_ents_with_attr

    print(f"The number of Wikidata entities that have an attribute triple: {len(wkd_ents_with_attr)}.")
    print(f"The number of Wikidata entities that have a linked entity in DBpedia: {len(wkd_linked_ents_set)}.")
    print(f"The number of Wikidata entities that do not have an attribute triple: {len(wkd_linked_ents_no_attr)}.")
    
    # Get the linked entity pairs where one of the entities DOES NOT HAVE an attribute triple, therefore the method
    # we are applying is inapplicable for these pairs.
    linked_ents_without_attr = get_linked_entity_pairs_with_no_attribute(dbp_linked_ents_no_attr,
                                                                         wkd_linked_ents_no_attr,
                                                                         linked_ents)
    
    # Get the linked entity pairs where both of the entities HAVE an attribute triple, therefore the method
    # we are applying is applicable for these pairs.
    linked_ents_with_attr = get_linked_entity_pairs_with_attribute(dbp_linked_ents_no_attr,
                                                                   wkd_linked_ents_no_attr,
                                                                   linked_ents)
    
    # For information purposes only, get the number of linked entity pairs where NEITHER entity has an
    # attribute. These are referred to as isolated pairs.
    get_isolated_linked_entity_pairs(dbp_linked_ents_no_attr,
                                     wkd_linked_ents_no_attr,
                                     linked_ents)
    
    # Create a new file of linked entity pairs that contains only those entity pairs where both entities have at
    # least one attribute triple.
    create_linked_entities_with_attribute_file(linked_ents_with_attr,
                                               dataset_path)
    
    # Create new lists of linked entities for DBpedia and Wikidata from the list containing linked entity pairs
    # where both entities have at least one attribute.
    dbp_linked_ents_v2, wkd_linked_ents_v2 = split_linked_entity_pairs(linked_ents_with_attr)
    
    # Check there are 13245 entities in each new split list.
    print(f"Number of entities in the new DBpedia list: {len(dbp_linked_ents_v2)}.")
    print(f"Number of entities in the new Wikidata list: {len(wkd_linked_ents_v2)}.")
    
    # Create two dictionaries:
    # 1. Keys - DBpedia entities; Values = Lists containing lists of [entitiy, attribute, value] for all associated
    # attribute triples.
    dbp_linked_ents_and_attributes = map_attribute_triples_to_entities(dbp_linked_ents_v2,
                                                                       dbp_ents,
                                                                       dbp_attr,
                                                                       dbp_vals)
    
    # 2. Keys - Wikidata entities; Values = Lists containing lists of [entitiy, attribute, value] for all associated
    # attribute triples.
    wkd_linked_ents_and_attributes = map_attribute_triples_to_entities(wkd_linked_ents_v2,
                                                                       wkd_ents,
                                                                       wkd_attr,
                                                                       wkd_vals)
    
    # Create a dictionary of format [Wikidata Property ID : Wikidata Property label] from the associated mapping file.\
    # This will be used to convert the Wikidata attributes that are written as property IDs into natural language.
    mapping_file = os.path.dirname(dataset_path) + "/wikidata_property_labels.csv"
    wkd_ids_to_title = utils.create_mapping_dictionary(mapping_file)
    
    # DBpedia
    # Create a new dictionary of format { entity : [attribute triples as sentences] }
    dbp_dict = dbp_convert_triples_to_sentencies(dbp_linked_ents_and_attributes,
                                                 sentence_type,
                                                 remove_value_data_type)
    
    try:
        with open(dbp_dict_pkl_path, 'wb') as dbp_dict_file:
            pickle.dump(dbp_dict, dbp_dict_file)
            print("Saved pickle file for DBpedia entities:attributes dictionary:" + os.path.abspath(dbp_dict_pkl_path))
            print("You won't need to preprocess for this dataset again!")
    except:
        print("Unable to save pickle file for DBpedia dictionary of entities and their associated attribute triples.")
    
    # Wikidata
    # Create a new dictionary of format { entity : [attribute triples as sentences] }
    wkd_dict = wkd_convert_triples_to_sentencies(wkd_linked_ents_and_attributes,
                                                 wkd_ids_to_title,
                                                 sentence_type,
                                                 remove_value_data_type)
    
    # Save out the dictionaries as pickle files so preprocessing is not required again for this
    # configuration.
    try:
        with open(wkd_dict_pkl_path, 'wb') as wkd_dict_file:
            pickle.dump(wkd_dict, wkd_dict_file)
            print("Saved pickle file for Wikidata entities:attributes dictionary:" + os.path.abspath(wkd_dict_pkl_path))
            print("You won't need to preprocess for this dataset again!")
    except:
        print("Unable to save pickle file for Wikidata dictionary of entities and their associated attribute triples.")
        
    return dbp_dict, wkd_dict