import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import re
import logging

# Try to import sentence_transformers, use fallback if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available, using TF-IDF only for semantic search")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalRetriever:
    """Medical question retrieval system using multiple similarity methods"""
    
    def __init__(self, processed_data: pd.DataFrame):
        self.data = processed_data
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.sentence_transformer = None
        self.sentence_embeddings = None
        self.initialize_retrievers()
    
    def initialize_retrievers(self):
        """Initialize TF-IDF and sentence transformer models"""
        try:
            # Initialize TF-IDF vectorizer
            logger.info("Initializing TF-IDF vectorizer...")
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            # Create TF-IDF matrix from combined text
            text_corpus = self.data['combined_text'].fillna('').tolist()
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_corpus)
            
            # Initialize sentence transformer if available
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.info("Loading sentence transformer model...")
                try:
                    self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                    
                    # Create sentence embeddings
                    logger.info("Creating sentence embeddings...")
                    questions = self.data['Question'].fillna('').tolist()
                    self.sentence_embeddings = self.sentence_transformer.encode(questions)
                except Exception as e:
                    logger.warning(f"Failed to initialize sentence transformer: {e}")
                    self.sentence_transformer = None
                    self.sentence_embeddings = None
            else:
                logger.info("Sentence transformers not available, using TF-IDF only")
                self.sentence_transformer = None
                self.sentence_embeddings = None
            
            logger.info("Retrieval system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing retrievers: {e}")
            raise
    
    def preprocess_query(self, query: str) -> str:
        """Preprocess user query for better matching"""
        # Convert to lowercase
        query = query.lower()
        
        # Remove common question patterns
        patterns_to_remove = [
            r'^(what is|what are|how to|can you tell me about|tell me about)',
            r'\?$',
            r'^(i want to know about|information about)',
        ]
        
        for pattern in patterns_to_remove:
            query = re.sub(pattern, '', query).strip()
        
        # Clean up extra spaces
        query = re.sub(r'\s+', ' ', query).strip()
        
        return query
    
    def tfidf_search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Search using TF-IDF similarity"""
        try:
            # Preprocess and vectorize query
            processed_query = self.preprocess_query(query)
            query_vector = self.tfidf_vectorizer.transform([processed_query])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top-k results
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            return [(idx, similarities[idx]) for idx in top_indices if similarities[idx] > 0.1]
            
        except Exception as e:
            logger.error(f"Error in TF-IDF search: {e}")
            return []
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Search using semantic similarity with sentence transformers"""
        try:
            # If sentence transformers not available, fall back to TF-IDF
            if self.sentence_transformer is None or self.sentence_embeddings is None:
                logger.info("Sentence transformers not available, using TF-IDF for semantic search")
                return self.tfidf_search(query, top_k)
            
            # Encode query
            query_embedding = self.sentence_transformer.encode([query])
            
            # Calculate cosine similarity with all question embeddings
            similarities = cosine_similarity(query_embedding, self.sentence_embeddings).flatten()
            
            # Get top-k results
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            return [(idx, similarities[idx]) for idx in top_indices if similarities[idx] > 0.3]
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def keyword_search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Search using keyword matching"""
        try:
            query_words = set(self.preprocess_query(query).split())
            scores = []
            
            for idx, row in self.data.iterrows():
                # Check focus, question, and answer for keyword matches
                text_fields = [
                    str(row['Focus']).lower(),
                    str(row['Question']).lower(),
                    str(row['Answer']).lower()
                ]
                
                combined_text = ' '.join(text_fields)
                text_words = set(combined_text.split())
                
                # Calculate overlap score
                overlap = len(query_words.intersection(text_words))
                total_query_words = len(query_words)
                
                if total_query_words > 0:
                    score = overlap / total_query_words
                else:
                    score = 0
                
                scores.append((idx, score))
            
            # Sort by score and return top-k
            scores.sort(key=lambda x: x[1], reverse=True)
            return [(idx, score) for idx, score in scores[:top_k] if score > 0.2]
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Comprehensive search combining multiple methods"""
        try:
            # Get results from different search methods
            tfidf_results = self.tfidf_search(query, top_k)
            semantic_results = self.semantic_search(query, top_k)
            keyword_results = self.keyword_search(query, top_k)
            
            # Combine and rank results
            all_results = {}
            
            # Add TF-IDF results with weight
            for idx, score in tfidf_results:
                all_results[idx] = all_results.get(idx, 0) + score * 0.3
            
            # Add semantic results with higher weight
            for idx, score in semantic_results:
                all_results[idx] = all_results.get(idx, 0) + score * 0.5
            
            # Add keyword results with moderate weight
            for idx, score in keyword_results:
                all_results[idx] = all_results.get(idx, 0) + score * 0.2
            
            # Sort by combined score
            sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
            
            # Format results
            formatted_results = []
            for idx, combined_score in sorted_results[:top_k]:
                if combined_score > 0.2:  # Minimum threshold
                    row = self.data.iloc[idx]
                    formatted_results.append({
                        'index': idx,
                        'focus': row['Focus'],
                        'question': row['Question'],
                        'answer': row['Answer'],
                        'category': row['Category'],
                        'question_type': row.get('question_type', 'general'),
                        'score': combined_score,
                        'semantic_type': row.get('SemanticType', ''),
                        'cui': row.get('CUI', ''),
                        'synonym': row.get('Synonym', '')
                    })
            
            logger.info(f"Found {len(formatted_results)} relevant results for query: {query}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive search: {e}")
            return []
    
    def search_by_category(self, query: str, category: str, top_k: int = 3) -> List[Dict]:
        """Search within a specific category"""
        try:
            # Filter data by category
            category_data = self.data[self.data['Category'] == category]
            
            if category_data.empty:
                return []
            
            # Create a temporary retriever for this category
            temp_retriever = MedicalRetriever(category_data)
            return temp_retriever.search(query, top_k)
            
        except Exception as e:
            logger.error(f"Error in category search: {e}")
            return []
    
    def get_similar_questions(self, question: str, top_k: int = 5) -> List[Dict]:
        """Find similar questions to the given question"""
        try:
            results = self.semantic_search(question, top_k)
            
            formatted_results = []
            for idx, score in results:
                row = self.data.iloc[idx]
                formatted_results.append({
                    'question': row['Question'],
                    'focus': row['Focus'],
                    'category': row['Category'],
                    'similarity_score': score
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error finding similar questions: {e}")
            return []
