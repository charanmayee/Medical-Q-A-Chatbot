import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedQuADProcessor:
    """Process and manage MedQuAD dataset"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.raw_data = None
        self.processed_data = None
        self.load_data()
        self.process_data()
    
    def load_data(self):
        """Load the MedQuAD CSV dataset"""
        try:
            self.raw_data = pd.read_csv(self.csv_path)
            logger.info(f"Loaded {len(self.raw_data)} records from MedQuAD dataset")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if pd.isna(text) or text == "":
            return ""
        
        # Convert to string and strip whitespace
        text = str(text).strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep medical punctuation
        text = re.sub(r'[^\w\s\-\.\,\?\!\(\)\/]', ' ', text)
        
        return text
    
    def process_data(self):
        """Process the raw data for better retrieval"""
        if self.raw_data is None:
            raise ValueError("No data loaded")
        
        # Create a copy for processing
        self.processed_data = self.raw_data.copy()
        
        # Clean text fields
        self.processed_data['Question'] = self.processed_data['Question'].apply(self.clean_text)
        self.processed_data['Answer'] = self.processed_data['Answer'].apply(self.clean_text)
        self.processed_data['Focus'] = self.processed_data['Focus'].apply(self.clean_text)
        
        # Remove rows with empty questions or answers
        self.processed_data = self.processed_data[
            (self.processed_data['Question'] != "") & 
            (self.processed_data['Answer'] != "")
        ]
        
        # Create combined text for better searching
        self.processed_data['combined_text'] = (
            self.processed_data['Focus'].fillna('') + ' ' +
            self.processed_data['Question'].fillna('') + ' ' +
            self.processed_data['Answer'].fillna('')
        ).apply(self.clean_text)
        
        # Extract question types
        self.processed_data['question_type'] = self.processed_data['Question'].apply(
            self.extract_question_type
        )
        
        # Create search-optimized question text
        self.processed_data['search_question'] = self.processed_data['Question'].apply(
            self.optimize_for_search
        )
        
        logger.info(f"Processed {len(self.processed_data)} valid Q&A pairs")
    
    def extract_question_type(self, question: str) -> str:
        """Extract the type of medical question"""
        if not question:
            return "general"
        
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['what is', 'what are']):
            return "definition"
        elif any(word in question_lower for word in ['how to prevent', 'prevention']):
            return "prevention"
        elif any(word in question_lower for word in ['treatment', 'how to treat']):
            return "treatment"
        elif any(word in question_lower for word in ['symptoms', 'signs']):
            return "symptoms"
        elif any(word in question_lower for word in ['causes', 'why']):
            return "causes"
        elif any(word in question_lower for word in ['risk', 'who is at risk']):
            return "risk_factors"
        elif any(word in question_lower for word in ['outlook', 'prognosis']):
            return "prognosis"
        elif any(word in question_lower for word in ['research', 'clinical trials']):
            return "research"
        else:
            return "general"
    
    def optimize_for_search(self, question: str) -> str:
        """Optimize question text for search"""
        if not question:
            return ""
        
        # Remove common question words that don't add meaning
        stop_words = [
            'do you have information about',
            'what is (are)',
            'what is the outlook for',
            'how to prevent',
            'who is at risk for',
            'do i need to see a doctor for',
            'what are the treatments for',
            'what research (or clinical trials) is being done for'
        ]
        
        optimized = question.lower()
        for stop_phrase in stop_words:
            optimized = optimized.replace(stop_phrase, '')
        
        # Clean up
        optimized = re.sub(r'\s+', ' ', optimized).strip()
        optimized = re.sub(r'[^\w\s]', ' ', optimized)
        
        return optimized
    
    def get_processed_data(self) -> pd.DataFrame:
        """Get the processed dataset"""
        return self.processed_data
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        if self.processed_data is None:
            return {}
        
        stats = {
            'total_questions': len(self.processed_data),
            'disease_count': len(self.processed_data[self.processed_data['Category'] == 'Disease']),
            'other_count': len(self.processed_data[self.processed_data['Category'] == 'Other']),
            'question_types': self.processed_data['question_type'].value_counts().to_dict(),
            'avg_question_length': self.processed_data['Question'].str.len().mean(),
            'avg_answer_length': self.processed_data['Answer'].str.len().mean()
        }
        
        return stats
    
    def search_by_focus(self, focus_term: str, limit: int = 10) -> List[Dict]:
        """Search questions by focus/topic"""
        if self.processed_data is None:
            return []
        
        focus_term = focus_term.lower()
        matches = self.processed_data[
            self.processed_data['Focus'].str.lower().str.contains(focus_term, na=False)
        ]
        
        return matches.head(limit).to_dict('records')
    
    def get_categories(self) -> List[str]:
        """Get all unique categories"""
        if self.processed_data is None:
            return []
        
        return self.processed_data['Category'].unique().tolist()
    
    def get_focuses(self) -> List[str]:
        """Get all unique focus areas"""
        if self.processed_data is None:
            return []
        
        return self.processed_data['Focus'].unique().tolist()
