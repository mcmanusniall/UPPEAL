import csv
import re

# This file contains utility functions.

# Returns the number of unique values from a list
def unique_value_count(my_list):
    return len(set(my_list))

# Returns a set of unique values from a list
def get_unique_values(my_list):
    return set(my_list)

# Counts the number of times each value appears in a list and returns a dictionary of
# format (value : count)
def count_elements(my_list):
    element_count = {}
    for element in my_list:
        if element in element_count:
            element_count[element] += 1
        else:
            element_count[element] = 1
    return element_count

# Sorts a dictionary containing values and their count in descending order
def sort_by_count_desc(dictionary):
    sorted_dict = dict(sorted(dictionary.items(), key=lambda item: item[1], reverse=True))
    return sorted_dict

# Creates a dictionary of mappings from a csv file containing two columns (to, from).
# This is used to map the property ID codes in Wikidata to their natural language title.
def create_mapping_dictionary(csv_file):
    print(f"Creating mapping dictionary from {csv_file}...")
    mapping_dict = {}
    with open(csv_file, newline='') as file:
        reader = csv.reader(file)
        next(reader, None)  # skip headers
        for row in reader:
            if len(row) == 2:
                key = row[0].strip()
                value = row[1].strip()
                mapping_dict[key] = value
            else:
                print(f"Invalid row format: {row}. Skipping this row.")
    
    print("Successful!")
    
    return mapping_dict

# Gets the resource from a URI
def get_resource_from_uri(uri):
    try:
        resource = uri.rsplit(sep = '/', maxsplit = 1)[1]
    except Exception as e:
        print(f"An error occurred when extracting the resource string from a URI: {e}") 

    return resource

# Converts a camel case string to a lowercase sentence.
# This is used to convert the DBpedia attributes into natural language.
def camel_case_to_sentence(string):
    words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', string)
    sentence = ' '.join(words).lower()

    return sentence

# Removes lines from a file.
def remove_lines_from_file(input_file, line_numbers_to_remove, output_file):
    try:
        with open(input_file, 'r') as file:
            lines = file.readlines()
            
        lines = [line for i, line in enumerate(lines, 1) if i not in line_numbers_to_remove]

        with open(output_file, 'w') as output_f:
            output_f.writelines(lines)

        print(f"{len(line_numbers_to_remove)} lines removed: {line_numbers_to_remove}.\nNew file created: '{output_file}'")
        
    except FileNotFoundError:
        print(f"Error: The input file '{input_file}' does not exist.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

def read_mapping_from_csv(csv_file):
    mapping = {}
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            mapping[row['original_value']] = row['mapped_value']
    return mapping

def replace_values_with_mapping(input_list, mapping):
    return [mapping.get(item, item) for item in input_list]