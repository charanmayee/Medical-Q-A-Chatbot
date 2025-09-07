import spacy
import re
from typing import List, Dict, Set
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalEntityRecognizer:
    """Medical entity recognition using spaCy and custom patterns"""
    
    def __init__(self):
        self.nlp = None
        self.medical_terms = set()
        self.initialize_models()
        self.load_medical_vocabulary()
    
    def initialize_models(self):
        """Initialize spaCy model"""
        try:
            # Try to load the English model
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.warning("spaCy English model not found. Using basic pattern matching.")
            self.nlp = None
    
    def load_medical_vocabulary(self):
        """Load medical terms and patterns"""
        # Common medical terms, symptoms, and conditions
        medical_vocab = {
            # Body parts
            'heart', 'lung', 'brain', 'liver', 'kidney', 'stomach', 'intestine',
            'blood', 'bone', 'muscle', 'skin', 'eye', 'ear', 'nose', 'throat',
            'chest', 'abdomen', 'pelvis', 'spine', 'joint', 'artery', 'vein',
            
            # Symptoms
            'pain', 'fever', 'headache', 'nausea', 'fatigue', 'dizziness',
            'cough', 'shortness of breath', 'swelling', 'inflammation',
            'bleeding', 'bruising', 'rash', 'itching', 'numbness', 'weakness',
            'confusion', 'memory loss', 'depression', 'anxiety', 'insomnia',
            
            # Conditions
            'diabetes', 'hypertension', 'cancer', 'infection', 'allergy',
            'asthma', 'pneumonia', 'bronchitis', 'arthritis', 'osteoporosis',
            'stroke', 'seizure', 'migraine', 'anemia', 'obesity', 'malnutrition',
            
            # Treatments
            'surgery', 'medication', 'therapy', 'treatment', 'procedure',
            'vaccination', 'immunization', 'antibiotics', 'chemotherapy',
            'radiation', 'transplant', 'rehabilitation', 'physical therapy',
            
            # Medical specialties
            'cardiology', 'neurology', 'oncology', 'dermatology', 'psychiatry',
            'orthopedics', 'gastroenterology', 'endocrinology', 'nephrology',
            'pulmonology', 'rheumatology', 'ophthalmology', 'otolaryngology'
        }
        
        self.medical_terms.update(medical_vocab)
        logger.info(f"Loaded {len(self.medical_terms)} medical terms")
    
    def extract_entities_with_spacy(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using spaCy NER"""
        if self.nlp is None:
            return {}
        
        try:
            doc = self.nlp(text)
            entities = {
                'conditions': [],
                'symptoms': [],
                'body_parts': [],
                'treatments': [],
                'medications': [],
                'organizations': [],
                'persons': []
            }
            
            for ent in doc.ents:
                entity_text = ent.text.lower()
                
                # Categorize entities based on their type and context
                if ent.label_ in ['DISEASE']:
                    entities['conditions'].append(ent.text)
                elif ent.label_ in ['ORG']:
                    entities['organizations'].append(ent.text)
                elif ent.label_ in ['PERSON']:
                    entities['persons'].append(ent.text)
                else:
                    # Use context clues to categorize
                    if any(term in entity_text for term in ['pain', 'ache', 'syndrome', 'disorder']):
                        entities['symptoms'].append(ent.text)
                    elif any(term in entity_text for term in ['therapy', 'treatment', 'surgery']):
                        entities['treatments'].append(ent.text)
                    elif any(term in entity_text for term in self.get_body_parts()):
                        entities['body_parts'].append(ent.text)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error in spaCy entity extraction: {e}")
            return {}
    
    def extract_entities_with_patterns(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using pattern matching"""
        entities = {
            'conditions': [],
            'symptoms': [],
            'body_parts': [],
            'treatments': [],
            'medications': [],
            'medical_terms': []
        }
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Define category patterns
        condition_patterns = [
            r'\b\w*itis\b',  # inflammation conditions
            r'\b\w*osis\b',  # disease conditions
            r'\b\w*emia\b',  # blood conditions
            r'\b\w*pathy\b', # disease conditions
            r'\b\w*syndrome\b', # syndromes
            r'\b\w*disorder\b', # disorders
        ]
        
        treatment_patterns = [
            r'\b\w*therapy\b',
            r'\b\w*surgery\b',
            r'\b\w*ectomy\b',  # surgical removals
            r'\b\w*plasty\b',  # surgical repairs
            r'\b\w*scopy\b',   # examinations
        ]
        
        # Extract patterns
        for pattern in condition_patterns:
            matches = re.findall(pattern, text_lower)
            entities['conditions'].extend(matches)
        
        for pattern in treatment_patterns:
            matches = re.findall(pattern, text_lower)
            entities['treatments'].extend(matches)
        
        # Extract known medical terms
        for word in words:
            if word in self.medical_terms:
                entities['medical_terms'].append(word)
                
                # Categorize known terms
                if word in self.get_body_parts():
                    entities['body_parts'].append(word)
                elif word in self.get_symptoms():
                    entities['symptoms'].append(word)
                elif word in self.get_treatments():
                    entities['treatments'].append(word)
        
        # Remove duplicates
        for category in entities:
            entities[category] = list(set(entities[category]))
        
        return entities
    
    def extract_entities(self, text: str) -> List[str]:
        """Main method to extract medical entities"""
        try:
            all_entities = []
            
            # Extract using spaCy if available
            if self.nlp is not None:
                spacy_entities = self.extract_entities_with_spacy(text)
                for category, ents in spacy_entities.items():
                    all_entities.extend(ents)
            
            # Extract using patterns
            pattern_entities = self.extract_entities_with_patterns(text)
            for category, ents in pattern_entities.items():
                all_entities.extend(ents)
            
            # Remove duplicates and filter out very short terms
            unique_entities = list(set([
                entity for entity in all_entities 
                if len(entity) > 2 and entity.lower() not in ['the', 'and', 'for', 'with']
            ]))
            
            logger.info(f"Extracted {len(unique_entities)} medical entities from text")
            return unique_entities
            
        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
            return []
    
    def get_detailed_entities(self, text: str) -> Dict[str, List[str]]:
        """Get detailed categorized entities"""
        try:
            # Combine results from both methods
            detailed_entities = {
                'conditions': [],
                'symptoms': [],
                'body_parts': [],
                'treatments': [],
                'medications': [],
                'medical_terms': []
            }
            
            if self.nlp is not None:
                spacy_entities = self.extract_entities_with_spacy(text)
                for category in detailed_entities:
                    if category in spacy_entities:
                        detailed_entities[category].extend(spacy_entities[category])
            
            pattern_entities = self.extract_entities_with_patterns(text)
            for category in detailed_entities:
                if category in pattern_entities:
                    detailed_entities[category].extend(pattern_entities[category])
            
            # Remove duplicates
            for category in detailed_entities:
                detailed_entities[category] = list(set(detailed_entities[category]))
            
            return detailed_entities
            
        except Exception as e:
            logger.error(f"Error in detailed entity extraction: {e}")
            return {}
    
    def get_body_parts(self) -> Set[str]:
        """Get body parts vocabulary"""
        return {
            'heart', 'lung', 'brain', 'liver', 'kidney', 'stomach', 'intestine',
            'blood', 'bone', 'muscle', 'skin', 'eye', 'ear', 'nose', 'throat',
            'chest', 'abdomen', 'pelvis', 'spine', 'joint', 'artery', 'vein',
            'head', 'neck', 'shoulder', 'arm', 'hand', 'leg', 'foot', 'back'
        }
    
    def get_symptoms(self) -> Set[str]:
        """Get symptoms vocabulary"""
        return {
            'pain', 'fever', 'headache', 'nausea', 'fatigue', 'dizziness',
            'cough', 'swelling', 'inflammation', 'bleeding', 'bruising',
            'rash', 'itching', 'numbness', 'weakness', 'confusion'
        }
    
    def get_treatments(self) -> Set[str]:
        """Get treatments vocabulary"""
        return {
            'surgery', 'medication', 'therapy', 'treatment', 'procedure',
            'vaccination', 'immunization', 'antibiotics', 'chemotherapy',
            'radiation', 'transplant', 'rehabilitation'
        }
