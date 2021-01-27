from typing import Dict, List, Tuple, Union
from pathlib import Path
from itertools import combinations
import logging

from pydantic import ValidationError
from flair.embeddings import TransformerDocumentEmbeddings, SentenceTransformerDocumentEmbeddings
from flair.data import Sentence
import torch
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from tqdm import tqdm
from urllib3.exceptions import MaxRetryError
from geopy.exc import GeocoderUnavailable

from models import Company
from utils import load_json, load_pickle, save_pickle

# DEBUGGING #
# import tracemalloc
# from utils import display_top_malloc_lines
#
# tracemalloc.start()
# --------- #

logger = logging.getLogger(__name__)

cos_similarity = torch.nn.CosineSimilarity(dim=0) 


class CompanyMatcher:
    companies: Dict[str, Company] = {}
    
    embedding: Union[TransformerDocumentEmbeddings, SentenceTransformerDocumentEmbeddings]
    description_embeddings: Dict[str, torch.tensor] = {}
    embeddings_pkl_file_path: Path = Path("description-embeddings.pkl")

    geolocator: Nominatim = Nominatim(user_agent="company-matcher")
    headquarters_locations: Dict[str, Tuple[float, float]] = {}
    locations_pkl_file_path: Path = Path("headquarters-locations.pkl")

    _similarity_component_weights: Dict[str, float] = {
        "description": 0.6, "founded": 0.2, "headquarters": 0.2
    }

    _founding_year_normalizer: int = None
    _hq_location_normalizer: int = 20000  # half of earth diameter
    # largest distance between two headquarters takes two long to compute
    # (at least with the inefficient method implemented in _calc_hq_location_normalizer

    def __init__(self, company_file_path: Path, transformer_model: str, sentence_transformer: bool = False,
                 similarity_component_weights: Dict[str, float] = None, load_n: int = None):
        self._load_transformer_model(transformer_model, sentence_transformer)
        self._load_companies(company_file_path, load_n)
        self._embed_descriptions()
        self._locate_headquarters()
        self._load_similarity_component_weights(similarity_component_weights)
        self._calc_founding_year_normalizer()

    def _load_similarity_component_weights(self, similarity_component_weights: Dict[str, float]):
        if similarity_component_weights:
            if {"description", "founded", "headquarters"} == set(self._similarity_component_weights.keys()):
                self._similarity_component_weights = similarity_component_weights
            else:
                logger.warning(f"Invalid similarity component weights gives: {similarity_component_weights}")
                logger.warning("Using default values!")

    def _load_transformer_model(self, transformer_model: str, sentence_transformer: bool):
        logger.info("Loading transformer model...")
        if sentence_transformer:
            try:
                self.embedding = SentenceTransformerDocumentEmbeddings(transformer_model, )
            except OSError as e:
                logger.error("Could not load transformer model: " + str(e))
                exit()
        else:
            try:
                self.embedding = TransformerDocumentEmbeddings(transformer_model, fine_tune=False)
            except OSError as e:
                logger.error("Could not load sentence transformer model: " + str(e))
                exit()
        logger.info("Done loading transformer model!")
    
    def _load_companies(self, company_file_path: Path, load_n: int): 
        logger.info("Loading company data from file...")
        try:
            json_data = load_json(company_file_path)
        except OSError as e:
            logger.error("Could not company data file: " + str(e))
            exit()
        
        if 0 < load_n <= len(json_data):
            json_data = json_data[:load_n]
        
        try:
            companies_list = [Company(**entry) for entry in json_data]
        except ValidationError as e:
            logger.error("Company data does not follow valid format: " + str(e))
            exit()
              
        try:
            companies_url_list = [c.url for c in companies_list]
            assert len(companies_url_list) == len(set(companies_url_list))
        except AssertionError:            
            logger.warning("Company URLs are not unique!")
        
        # check which which companies are duplicates
        duplicate_company_urls = []
        for company in companies_list:
            if self.companies.get(company.url, None):
                duplicate_company_urls.append(company.url)
            else:
                self.companies[company.url] = company
        logger.warning(f"Following company URLs have multiple entries: {duplicate_company_urls}")
        logger.warning("Duplicate entries will be ignored!")
        
        logger.info("Done loading company data!")
    
    def _embed_descriptions(self, chunk_size: int = 30, load_from_pickle: bool = True, save_to_pickle: bool = True):
        if load_from_pickle:
            self.description_embeddings = load_pickle(self.embeddings_pkl_file_path,
                                                      error_msg="Could not load stored embeddings!")
        
        descriptions_ = [(company.url, Sentence(company.description)) 
                         for company in self.companies.values() if company.url not in self.description_embeddings]
        
        # chunking for progress bar
        if descriptions_:
            logger.info("Computing description embeddings...")
            
            with tqdm(total=len(descriptions_)) as pbar:
                for start_idx in range(0, len(descriptions_), chunk_size):
                    end_idx = start_idx + chunk_size
                    if not end_idx < len(descriptions_):
                        end_idx = len(descriptions_)
                        chunk_size = end_idx - start_idx

                    descriptions_chunk = descriptions_[start_idx:end_idx]
                    self.embedding.embed([description_[1] for description_ in descriptions_chunk])
                    self.description_embeddings.update({description_[0]: description_[1].embedding
                                                        for description_ in descriptions_chunk})
                    # remove embedding from sentence objects
                    for _, description_sentence in descriptions_chunk:
                        description_sentence.clear_embeddings()

                    if save_to_pickle:
                        save_pickle(object_=self.description_embeddings, pkl_file_path=self.embeddings_pkl_file_path,
                                    error_msg="Could not save new embeddings!")

                    pbar.update(chunk_size)
                    
                    # DEBUGGING #
                    # snapshot = tracemalloc.take_snapshot()
                    # display_top_malloc_lines(snapshot)
                    # --------- #

            logger.info("Done computing description embeddings!")

    def _locate_headquarters(self, load_from_pickle: bool = True, save_to_pickle: bool = True):
        if load_from_pickle:
            self.headquarters_locations = load_pickle(self.locations_pkl_file_path,
                                                      error_msg="Could not load stored locations!")
        
        not_located_companies = [company_url for company_url in self.companies
                                 if company_url not in self.headquarters_locations]

        not_located_companies = []  # DEBUGGING

        if not_located_companies:
            logger.info("Geo-locating company headquarters...")

            for company_url in not_located_companies:
                company = self.companies[company_url]
                if company.headquarters and not self.headquarters_locations.get(company_url, None):
                    location = None
                    try:
                        location = self.geolocator.geocode(company.headquarters)
                    except (MaxRetryError, GeocoderUnavailable):
                        pass
                    if location:
                        self.headquarters_locations[company_url] = (location.latitude, location.longitude)

                        if save_to_pickle:
                            save_pickle(object_=self.headquarters_locations, pkl_file_path=self.locations_pkl_file_path,
                                        error_msg="Could not save locations!")
            
            logger.info("Done locating company headquarters!")

    def _calc_founding_year_normalizer(self):
        company_founding_years = [company.founded for company in self.companies.values() if company.founded]
        min_founding_year = min(company_founding_years)
        max_founding_year = max(company_founding_years)
        self._founding_year_normalizer = max_founding_year - min_founding_year

    # function is to inefficient for datasets with more than a few hundred entries
    # def _calc_hq_location_normalizer(self):
    #     location_pairs = combinations(self.headquarters_locations.values(), 2)
    #     location_distances = [geodesic(location_pair[0], location_pair[1]).kilometers
    #                           for location_pair in location_pairs]
    #     self._hq_location_normalizer = max(location_distances)

    # -----------------------------#
    # --- Similarity functions --- #

    def _description_similarities(self, query_company: Company):
        query_embedding = self.description_embeddings[query_company.url]
        
        return {candidate_url: float(cos_similarity(query_embedding, candidate_embedding))
                for candidate_url, candidate_embedding in self.description_embeddings.items()
                if candidate_url != query_company.url}
    
    def _founded_similarities(self, query_company: Company):
        return {company.url: self._calc_founded_similarity(query_company, company)
                for company in self.companies.values() if company.url != query_company.url}
        
    def _calc_founded_similarity(self, query_company: Company, candidate_company: Company) -> Union[float, None]:
        if query_company.founded and candidate_company.founded:
            return 1 - abs(query_company.founded - candidate_company.founded) / float(self._founding_year_normalizer)
        else:
            return None

    def _headquarters_similarities(self, query_company: Company):
        return {company.url: self._calc_headquarters_similarity(query_company, company)
                for company in self.companies.values() if company.url != query_company.url}

    def _calc_headquarters_similarity(self, query_company: Company, candidate_company: Company) -> Union[float, None]:
        query_location = self.headquarters_locations.get(query_company.url, None)
        candidate_location = self.headquarters_locations.get(candidate_company.url, None)
        
        if query_location and candidate_location:
            return 1 - geodesic(query_location, candidate_location).kilometers / float(self._hq_location_normalizer)
        else:
            return None

    def get_peers(self, query_url: str, top_k: int = 10) -> Union[List[str], None]:
        if query_url not in self.companies:
            logger.warning(f"Company with URL '{query_url}' does not exist!")
            return None
        
        query_company = self.companies[query_url]

        description_similarities = self._description_similarities(query_company)
        founded_similarities = self._founded_similarities(query_company)
        headquarters_similarities = self._headquarters_similarities(query_company)
        
        similarities = []
        for company_url in description_similarities:
            if founded_similarities[company_url] and headquarters_similarities[company_url]:
                similarity = (self._similarity_component_weights['description'] * description_similarities[company_url]
                              + self._similarity_component_weights['founded'] * founded_similarities[company_url]
                              + self._similarity_component_weights['headquarters'] * headquarters_similarities[company_url])
            
            elif founded_similarities[company_url] and not headquarters_similarities[company_url]:
                weight_normalizer = self._similarity_component_weights['description'] + self._similarity_component_weights['founded'] 
                similarity = (self._similarity_component_weights['description'] * description_similarities[company_url]
                              + self._similarity_component_weights['founded'] * founded_similarities[company_url]) / float(weight_normalizer)

            elif not founded_similarities[company_url] and headquarters_similarities[company_url]:
                weight_normalizer = self._similarity_component_weights['description'] + self._similarity_component_weights['headquarters'] 
                similarity = (self._similarity_component_weights['description'] * description_similarities[company_url]
                              + self._similarity_component_weights['headquarters'] * headquarters_similarities[company_url]) / float(weight_normalizer)
            
            else:    
                weight_normalizer = self._similarity_component_weights['description'] \
                    + self._similarity_component_weights['founded'] + self._similarity_component_weights['headquarters'] 
                similarity = description_similarities[company_url] / float(weight_normalizer)    
            
            similarities.append((company_url, similarity))

        sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        return [x[0] for x in sorted_similarities][:top_k]

