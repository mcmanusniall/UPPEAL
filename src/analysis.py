import logging
import os

# This file contains generic functions for analysing the DBP-WD-15K datasets.

# Checks for attribute triples that do not have a value.
# Accepts a file containing RDF formatted triples.
def check_null_values(filename):
    try:
        with open(filename, 'r') as file:
            to_check = []
            lines = file.readlines()
            line_number = 0
            
            for line in lines:
                line_number += 1
                try:
                    entity, attribute, value = line.strip().split(maxsplit=2)
                except ValueError as e:
                    to_check.append([line_number, line, e])
                except Exception as e:
                    to_check.append([line_number, line, e])
                
    except FileNotFoundError:
        print(f"Error: The file '{filename}' could not be found.")
        return None

    except Exception as e:
        print(f"Error: An error occurred while reading the file: {e}")
        return None
    
    return to_check

# Accepts a dictionary of format {entity : [attribute triples]} and returns a dictionary
# of format {entity : number of associated attribute triples}.
# The dictionary is sorted in descending order.
def get_count_of_attributes_per_entity(linked_entities_and_attributes_dict):
    counts_per_entity = {}
    for entity, attribute_triples in linked_entities_and_attributes_dict.items():
        counts_per_entity[entity] = len(attribute_triples)
    
    sort_descending = sorted(counts_per_entity.items(), key=lambda x: x[1], reverse=True)    
    return sort_descending

# Returns the total number of attribute triples included in a dictionary of format
# {entity : [attribute triples]}.
def get_sentence_count_from_dictionary(dictionary):
    sentences = 0
    for key, tuples in dictionary.items():
        for tuple in tuples:
            sentences += 1

    return sentences