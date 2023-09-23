import tenseal as ts
import os

# This file manages the encryption of embeddings using the CKKS
# Fully Homomorphic Encryption scheme.
# Please refer to the Tenseal GitHub page for more information
# https://github.com/OpenMined/TenSEAL

# Creates a Tenseal context object. This stores the parameters of the
# encryption scheme.
def context():
    print("Creating CKKS configuration...")
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes = [60, 40, 40, 60]
    )
    context.generate_galois_keys()
    context.global_scale = 2**40
    print("CKKS configured.")
    return context

# Encrypts a vector of arbitrary length using the context passed as an argument.
# Returns a Tenseal CKKSVector object.
def encrypt(vector, context):
    enc = ts.ckks_vector(context, vector)
    return enc

# Encrypts a dictionary of the format {entity : [embedding]}.
# Return a new dictionary of the format {entity : CKKSVector}.
def encrypt_dictionary_of_embeddings(dictionary, context):
    print("Encrypting embeddings...")
    enc_dictionary = {}
    for entity, embedding in dictionary.items():
        enc_vec = encrypt(embedding, context)
        enc_dictionary[entity] = enc_vec
    print("Complete!")
    return enc_dictionary