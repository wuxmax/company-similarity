import logging
from pathlib import Path

from matcher import CompanyMatcher

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# path to JSON file with company data
FILE_PATH = "company_collection.json"

# name of pre-trained transformer model. this can be a ...
# huggingface model: https://huggingface.co/transformers/pretrained_models.html
# SentenceTransformer model: https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/
TRANSFORMER_MODEL = "albert-large-v2"
# TRANSFORMER_MODEL = "roberta-large"
# TRANSFORMER_MODEL = "stsb-roberta-large" # this is a SentenceTransformer model

# if SentenceTransformer model is chosen, this must be 'True'
SENTENCE_TRANSFORMER = False

# load only N first items of dataset, loads all if set to <=0
LOAD_N = -1

# weights for the different components of the similarity function
SIMILARITY_COMPONENT_WEIGHTS = {
        "description": 0.6, "founded": 0.2, "headquarters": 0.2
}

TEST_QUERIES = ["farmlogs.com", "kuka.com", "glassbeam.com", "sigfox.com", "slantrange.com",
               "wavebl.com", "6river.com", "bionichive.com","2getthere.eu", "theyield.com"]
RUN_TEST_QUERIES = False


def run_test_queries(company_matcher: CompanyMatcher):
    print(f"{40 * '-'}")
    for query_url in TEST_QUERIES:
        print(f"{query_url:<20}: {company_matcher.get_peers(query_url)}")


def run():
    company_matcher = CompanyMatcher(transformer_model=TRANSFORMER_MODEL, company_file_path=Path(FILE_PATH),
                                     sentence_transformer=SENTENCE_TRANSFORMER, load_n=LOAD_N,
                                     similarity_component_weights=SIMILARITY_COMPONENT_WEIGHTS)

    if RUN_TEST_QUERIES:
        run_test_queries(company_matcher)

    while True:
        print(f"{40*'-'}")
        query_url = input("Enter URL of company: ")

        most_similar_companies = company_matcher.get_peers(query_url)

        if most_similar_companies:
            query_company = company_matcher.companies[query_url]
            print(f"{query_company.url:<20}| {str(query_company.founded):<4} | {str(query_company.headquarters):<40} | "
                  f"{query_company.description}")
            
            print(f"{20*'-'}\nMost similar companies:")
            for company_url in most_similar_companies:
                company = company_matcher.companies[company_url]
                print(f"{5*'-'}")
                print(f"{company.url:<20}| {str(company.founded):<4} | {str(company.headquarters):<40} | {company.description}")
            

if __name__ == "__main__":
    run()

