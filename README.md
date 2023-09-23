Author: Niall McManus (nmcman01@student.bbk.ac.uk)

Welcome to ‘**Privacy-Preserving Unsupervised Entity Alignment for Heterogeneous Knowledge Graphs using Pre-Trained Language Models and Homomorphic Encryption**’.

First, please have a look at the associated paper ‘*Privacy-Preserving Unsupervised Entity Alignment for Heterogeneous Knowledge Graphs using Pre-Trained Language Models and Homomorphic Encryption*’ by McManus (2023). The motivation behind this project and how to run this package are explained below, but this paper provides the technical detail behind the approach. **If you simply want to execute this code, then please refer to the section named ‘How do I run this code?’ at the end of this file.**

*Note: This package was developed using Python 3.10.12 on a Microsoft Azure ‘Standard_NC4as_T4_v3’ virtual machine with 4 AMD EPYC 7V12 (Rome) vCPUs, 28GiB RAM and 16GiB NVIDIA T4 GPU, and Linux Ubuntu 22.04.*

**Project Details.**

This project’s primary goal was to develop an embedding-based entity alignment (EA) method that can be conducted in an unsupervised, privacy-preserving manner. Currently, no known EA method exists that can be applied to privacy-critical settings, whereby two parties are required to conduct entity alignment between their respective knowledge graphs (KGs) while ensuring that the opposing party can never access the underlying data of their graph.

This package contains a full unsupervised, privacy-preserving EA pipeline that mimics the scenario described above. The solution leverages two pre-trained language models, *All-MPNet-Base-v2* (<https://huggingface.co/sentence-transformers/all-mpnet-base-v2>) and *Sentence-T5-Large* (<https://huggingface.co/sentence-transformers/sentence-t5-large>) to create text embeddings from an entity’s associated attribute triples. These embeddings are pooled to derive a single entity embedding, which is then encrypted using the CKKS fully homomorphic encryption scheme to protect it from inversion attacks. Based on the assumption that similar entities will have similar attributes, the similarity between entity embeddings is computed using the homomorphic dot product similarity function. The most similar entity is selected as the potential match that refers to the same real-world object.

This solution has been tailored to work with the benchmark EA dataset – *DBP-WD-15K* (details of this dataset can be found in the paper ‘*A Benchmarking Study of Embedding-based Entity Alignment for Knowledge Graphs*’ by Sun et al. (2020), or here - <https://github.com/nju-websoft/OpenEA>). This dataset contains the attribute triples for 15,000 entities extracted from the real-world *DBpedia* and *Wikidata* KGs. This dataset also contains a mapping file that defines the pairs of entities from each KG that refer to the same real-world object, allowing the performance of the solution to be evaluated.

Unfortunately, the performance is unsatisfactory, with the best-performing model configuration achieving a one-to-one (i.e., hits@1) matching rate of only 14.9%.

This package may be freely copied and modified.

**What does embedding-based mean?**

This refers to the way this solution models entities – as numeric vector representations known as embeddings. To derive an entity embedding, the text-based attributes associated with an entity are passed through a pre-trained language model to generate text embeddings. These text embeddings are then aggregated using a mean pooling operation to derive entity embeddings that can be compared to determine if two entities in different knowledge graphs refer to the same real-world object or concept.

**What does unsupervised mean?**

Generally, current EA methods transform entity embeddings into a unified vector space using seed alignments – pre-aligned pairs of entities. They train a neural network using these pre-aligned ground-truth matches as supervision. The network learns a transformation function that makes entity embeddings of matching entities have similar values. Once trained, all other embeddings are passed through this neural network so that the embeddings are better suited for the EA task. Pre-aligned entity pairs cannot be gathered in privacy-critical settings, as it would require exposing the underlying data. Therefore, this approach doesn’t require seed alignments or the transformation of embeddings. This is because the attribute-derived text embeddings generated from pre-trained language models already belong to the same vector space and can be compared as they are.

**What does privacy-preserving mean?**

This is main motivation behind this project – to develop an EA method that can be leveraged in privacy-critical settings. This means that two parties can find matching entities between their KGs without exposing any data to the opposing party. This is achieved by making the method unsupervised, as described above. Additionally, as text embeddings are susceptible to inversion attacks, another layer of security is added by encrypting entity embeddings using the CKKS homomorphic encryption scheme.

**What is homomorphic encryption?**

Homomorphic encryption is an encryption scheme that enables mathematical operations to be conducted on encrypted data. This enables the dot product similarity to be calculated on encrypted embeddings, therefore enabling EA while ensuring the underlying data of both KGs remains secure.

**What you need to know about this package.**

**The ‘**data**’ folder.**

The ‘data’ folder contains the benchmark ‘*DBP-WD-15K*’ datasets. There are two datasets included, ‘D_W_15K_V1’ and ‘D_W_15K_V2’. They both contain RDF formatted attribute triples for 15,000 entities sampled from two real-world KGs – *DBpedia* and *Wikidata* – and a file containing the pairs of entities from each graph that refer to the same real-world object.

D_W_15K_v1 – in this dataset, the distribution of entities and attributes is equivalent to the original KGs.

D_W_15K_v2 – this dataset contains a more synthetic representation of *DBpedia* and *Wikidata*. There are slightly more attribute triples per entity.

There are multiple files in each dataset:

-   attr_triples_1 - contains the RDF attribute triples extracted from DBpedia.

-   attr_triples_2 - contains the RDF attribute triples extracted from Wikidata.

-   ent_links - contains the matching pairs of entities from both KGs.

-   attr_triples_1_clean – this is a processed version of attr_triples_1 such that attribute triples with null values have been removed.

-   attr_triples_2_clean – this is a processed version of attr_triples_2 such that attribute triples with null values have been removed.

-   ent_links_clean – this is a processed version of ‘ent_links’ that contains only entity pairs where both entities have at least one associated attribute triple. As this solution works solely with attribute triples, if an entity does not have an attribute triple, no embedding can be generated. Hence, the creation of this file.

-   \*.pkl – these are pickle files that contain Python dictionary objects of the format { entity : [processed_attribute_triples] }. This solution offers multiple ways to process attribute triples prior to embedding. Each combination of these pre-processing methods has already been executed and the returned dictionary objects saved in these pickle files to save users from pre-processing the files again. If you would like to pre-process the data from scratch, please remove these .pkl files and the \_clean files.

-   wikidata_property_labels.csv – as Wikidata RDF triples specify attribute types as unique identifiers, this file contains mapping that transform them into their natural language representation.

**The ’**src**’ folder.**

This folder contains the source code for this solution. The below provides a description of the category of functions contained in each file. For a description of each function, please refer to comments above each function in the respective file.

-   analysis.py – contains functions for analysing the attribute triple files.

-   embedding.py – contains functions for generating text embeddings from attribute triples. Two pre-trained language models are available for performing embedding- *All-MPNet-Base-v2* and *Sentence-T5-Large*. These are implemented using the sentence-transformers library (https://www.sbert.net/).

-   encryption.py – contains functions for encrypting entity embeddings using the CKKS homomorphic encryption scheme. This has been implemented using the Tenseal library (https://github.com/OpenMined/TenSEAL).

-   inference.py – contains functions for computing the similarity between embeddings. This package may be run with or without encryption applied. As such there are two separate dot product similarity functions included in this file – one for encrypted embeddings and one for unencrypted embeddings. Please note, there is a second function for computing the dot product similarity between encrypted embeddings that is commented out. This is the function that is replicative of how this solution would be applied in a real-world scenario, as it computes the homomorphic dot product similarity. This is extremely computationally intensive and is not recommended to be executed. The uncommented version of this function implements a faster method that still allows evaluation of the effects of encryption with regards to the performance of the solution. This cannot be applied in a real-world scenario, as it encrypts and then decrypts the embeddings prior to computing the similarity. This still introduces the approximation of values that occurs when encrypting embeddings using CKKS but avoids performing the computationally expensive, homomorphic dot product similarity.

-   main.py – the entire program executes from this file. Details on how to configure a Python environment and run this file from the CLI with customisable parameters is included below.

-   metrics.py – contain functions for two evaluation metrics: *hits@k* and *mean reciprocal rank (MRR)*. This is how the performance of the method is evaluated. Details of these metrics are included in the associated paper.

-   pool_and_normalise.py – contains functions for performing mean pooling on multiple attribute-derived text embeddings to generate a single entity embedding and functions for normalising entity embeddings (i.e., transforming them into unit vectors).

-   pre-processing.py – contains functions for pre-processing the attribute triple files and entity links file. As mentioned above, these functions will not be executed if the \*.pkl files exist. There are four customisable techniques for pre-processing attribute triples, of which multiple combinations can be used. These techniques are as follows. Examples of how to execute different configurations of these techniques are included below in ‘*How do I run this code?*’.

-   *Technique 1 -* Removing the *subject* (i.e., entity) from an attribute triple

-   *Technique 2 -* Removing the attribute value *type* from attribute values.

-   *Technique 3 -* Concatenating all attribute values for a given entity into one string.

-   *Technique 4 -* Concatenating all attribute types and values for a given entity into one string.

Please note, when *Technique 3 or 4* are applied there is only one string to embed, therefore mean pooling is not performed. The single text embedding simply gets normalised and assigned as the entity embedding.

-   utils.py – this file contains general functions.

**How do I run the code?**

To run this code, it is strongly recommended that a new Python virtual environment is configured to install package dependencies and execute the program. It is also advised to run this code on a machine equipped with an NVIDIA GPU since it can notably expedite embedding and inference, though this is not a requirement as the program is also compatible with CPU architectures. Please note, if using an NVIDIA GPU equipped machine, you will be required to install the relevant version of the Python package cudatoolkit. This will be specific to the NVIDIA GPU driver and CUDA version of your machine.

Below are the required steps to configure a new Anaconda environment, install package dependencies, and execute the program on a Windows 10 machine. It is assumed that Anaconda is installed on the client machine as the Python virtual environment manager and pip is installed as the package manager. It is also assumed that the client machine is internet connected. This is required to download and install all dependencies.

The following package dependencies are required:

numpy==1.23.1

sentence-transformers==2.2.2

tenseal==0.3.14

torch==2.0.1

scikit-learn==1.3.0

The following steps have been tested and are confirmed to work on Birkbeck’s Department for Computer Science (DCS) laboratory computers. These machines also satisfy the above assumptions (i.e., they have Anaconda installed and are internet connected).

1.  First, clone this repository or download the .zip file (if downloading the .zip file, ensure that the package contents are extracted.

2.  Open the Command Prompt.

1.  Create a new Anaconda environment named ‘UPPEAL’ with Python 3.10.12 as the interpreter version:

conda create -n UPPEAL python==3.10.12

1.  Activate the new environment:

    conda activate UPPEAL

2.  Install NumPy (v1.23.1):

pip install numpy==1.23.1

1.  Install Sentence-Transformers (v2.2.2):

pip install sentence-transformers==2.2.2

1.  Install Tenseal (v0.3.14):

pip install tenseal==0.3.14

1.  Install PyTorch (v2.0.1):

pip install torch==2.0.1

1.  Install Scikit-Learn (v1.3.0):

pip install scikit-learn==1.3.0

1.  Change directory to the source folder “../UPPEAL/src” (angle brackets are not required):

    cd /d \<path_to_the_package\\UPPEAL\\src\>

2.  To run the program with the default parameters:

python main.py

1.  There are several command line arguments that can be specified to run the program with different parameter values/model configurations. These are as follows:

    \--dataset : this flag specifies what dataset the EA method is to be performed on. The values accepted are either “D_W_15K_V1” or “D_W_15K_V1”.

    \--model_name : this flag specifies what pre-trained language model is used for performing text embedding of attribute triples. The values accepted are either “all-mpnet-base-v2” or “sentence-t5-large”.

    \--sentence_type : this flag specifies what pre-processing techniques are applied to attribute triples. The values accepted are “attr_val” – this removes the entity from an attribute triple; “attr_val_one_sentence” – this removes the entity from an attribute triple and concatenates all attribute types and values for an entity into one string; “val_one_sentence” – this removes the entity and the attribute type from an attribute triple and concatenates all attribute values into one string.

    \--remove_value_data_type : include this flag to remove the data type from an attribute value.

    \--element_wise_op : this flag specifies the pooling operation to be performed on attribute embeddings. The values accepted are “mean”, to denote mean pooling, or “normalise_only”, to denote that no pooling operation is required. If “attr_val_one_sentence” and “val_one_sentence” are specified as the –-sentence_type, this argument will default to “normalise_only”.

    \--evaluation_direction : this flag specifies if DBpedia entities should be compared with Wikidata entities or vice versa. The values accepted are “dbp_to_wkd” for DBpedia to Wikidata, or “wkd_to_dbp” for Wikidata to DBpedia.

    \--enc : include this flag to encrypt entity embeddings using the CKKS fully homomorphic encryption scheme. Please note, this increases the computational cost of the method exponentially.

**Examples.**

Below are some examples of how these flags can be utilised.

*Example 1.*

Perform entity alignment on the *DBP-WD-15K-V1* dataset. Perform embedding using *Sentence-T5-Large*. Pre-process attribute triples to keep only the attribute values and concatenate them into one sentence. Remove the attribute value data type. Compare DBpedia entities with Wikidata entities. Do not encrypt the embeddings.

python main.py --dataset "D_W_15K_V1" --model_name "sentence-t5-large" --sentence_type "val_one_sentence" --remove_value_data_type --element_wise_op "normalise_only" --evaluation_direction "dbp_to_wkd"

*Example 2.*

Perform entity alignment on the *DBP-WD-15K-V2* dataset. Perform embedding using *All-MPNet-Base-v2*. Pre-process attribute triples to keep only the attribute types and values. Perform mean pooling to derive single entity embeddings. Do not remove the attribute value data type. Compare DBpedia entities with Wikidata entities. Encrypt the embeddings.

python main.py --dataset "D_W_15K_V2" --model_name "all-mpnet-base-v2" --sentence_type "attr_val" --element_wise_op "mean" --evaluation_direction "wkd_to_dbp" --enc

*Default Parameters.*

Perform entity alignment on the *DBP-WD-15K-V1* dataset. Perform embedding using *Sentence-T5-Large*. Pre-process attribute triples to keep only the attribute values and concatenate them into one sentence. Do not remove the attribute value data type. Compare DBpedia entities with Wikidata entities. Do not encrypt the embeddings.

python main.py

The arguments you have specified will be returned to the command prompt window upon execution:

dataset: D_W_15K_V1

model_name: sentence-t5-large

sentence_type: val_one_sentence

remove_value_data_type: False

element_wise_op: mean

evaluation_direction: dbp_to_wkd

enc: False

Once the model has successfully executed, the evaluation metrics will be returned. These include the *hits@1, hits@5, hits@10,* and *hits@50* scores, as well as the *Mean Reciprocal Rank (MRR)*. The output will look similar to the below.

Total Hits @ 1: 1373.

Percentage of Hits @ 1: 0.10366175915439789.

Total Hits @ 5: 1947.

Percentage of Hits @ 5: 0.14699886749716876.

Total Hits @ 10: 2206.

Percentage of Hits @ 10: 0.16655341638354096.

Total Hits @ 50: 2911.

Percentage of Hits @ 50: 0.21978104945262364.

Computing MRR...

Mean Reciprocal Rank (MRR) = 0.1261

Execution_time = 808.5615735054016s.
