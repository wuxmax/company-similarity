# Company Similarity Matcher

This simple python tool outputs the companies, which are most similar to a queried company in a given dataset.

# Installation
To use the tool, install the required python packages (preferably in a virtual environment):
```
pip install -r requirements.txt
```

# Usage
Run the program from this folder with the command:

```
python app/main.py
```

This expects a `company_collection.json` file to be in this folder as well.

# Data
Entries in the JSON file should have the following format:
```
{
    "name": "Example Company",
    "founded": "2000",
    "url": "example.com",
    "headquarters": "Example Country, Example State, Example City",
    "description": "Example company does anything."
}
```

Values for fields `founded` and `headquarters` can be empty, values for other fields must be present.

# Customization
The behaviour of the tool can be customized by setting the following variables in the `main.py` file:
```
# path to JSON file with company data
FILE_PATH = "./company_collection.json"

# name of pre-trained transformer model. this can be a ...
# huggingface model: https://huggingface.co/transformers/pretrained_models.html
# SentenceTransformer model: https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/
TRANSFORMER_MODEL = "albert-large-v2"
# TRANSFORMER_MODEL = "stsb-roberta-large" # this is a SentenceTransformer model

# if SentenceTransformer model is chosen, this must be 'True'
SENTENCE_TRANSFORMER = False

# load only N first items of dataset, loads all if set to <=0
LOAD_N = -1

# weights for the different components of the similarity function
SIMILARITY_COMPONENT_WEIGHTS = {
        "description": 0.6, "founded": 0.2, "headquarters": 0.2
}
```

# Chosen approach and possible improvements
This tool only implements a very basic approach. The idea is to leverage all of the data fields, which (in my personal opinion) promise to contain information significant for a similarity measure. This fields are: `descripion`, `founded` and `headquarters`.

* __description__:
The description of the company probably contains the most information about it, but at the same time the information is also the least structured.  
My approach is to generate embeddings of the description using the `flair` library and using a pre-trained [huggingface transformers](https://huggingface.co/transformers/pretrained_models.html) or [SentenceTransformers](https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0) model. For my testing I chose the `albert-large-v2` huggingface model, because it is an autoencoding model which can compete with the state-of-the-art. Autoencoders are (AFAIK) better suited for similarity tasks as e.g. autogressors. The similarity of the computed embeddings of the company descriptions is simply determined using cosine similarity.  
Probably one would get better embeddings for this task, if the language model would have been fine-tuned on the domain data. Also the method which is used by `flair` to generate the document embeddings might not be ideal for this task. Using a `SentenceTransformer` model might also help capturing semantics of the description (this possibility is already implemented, why I decided against it see **Known limitations**).  
More information could be leveraged from the descriptions by e.g. by extracting products or economic sectors from them. This could be done with an NER model which has been trained for this kind of entities.

* __founded__:
The idea here is that companies whose founding date is closer are more similar. Therefore the similarity of two companies is determined by the absolute difference of their founding years normalized by the greatest difference in the dataset.

* __headquarters__:
The same idea is applied as for the `founded` field: Two companies, whose headquarters are closer are more similar. To get the coordinate location of the headquarters, the `geopy` package is used.

The weight which each field has for the final similarity measure, can be changed by modifying the `SIMILARITY_COMPONENT_WEIGHTS`.

## Example results:






