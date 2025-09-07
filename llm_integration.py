import json
import os
import sys
from typing import List, Dict, Optional
import logging

# Import OpenAI
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalLLM:
    """LLM integration for enhanced medical responses"""
    
    def __init__(self):
        # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
        # do not change this unless explicitly requested by the user
        self.model = "gpt-5"
        self.client = None
        self.initialize_client()
    
    def initialize_client(self):
        """Initialize OpenAI client"""
        try:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not found in environment variables")
                return
            
            self.client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            self.client = None
    
    def generate_response(self, question: str, retrieval_results: List[Dict], 
                         entities: List[str]) -> str:
        """Generate enhanced response using LLM"""
        if self.client is None:
            return self.fallback_response(question, retrieval_results)
        
        try:
            # Prepare context from retrieval results
            context = self.prepare_context(retrieval_results)
            
            # Create system prompt
            system_prompt = self.create_system_prompt()
            
            # Create user prompt
            user_prompt = self.create_user_prompt(question, context, entities)
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.3,
                top_p=0.9
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return self.fallback_response(question, retrieval_results)
    
    def create_system_prompt(self) -> str:
        """Create system prompt for medical AI assistant"""
        return """You are a knowledgeable medical AI assistant. Your role is to provide helpful, accurate, and well-informed responses to medical questions based on reliable medical data.

IMPORTANT GUIDELINES:
1. Always emphasize that your responses are for informational purposes only
2. Recommend consulting healthcare professionals for personalized medical advice
3. Base your responses on the provided medical database information when available
4. If the database doesn't contain relevant information, provide general medical knowledge while being clear about limitations
5. Use clear, understandable language while maintaining medical accuracy
6. Include relevant medical terminology but explain complex terms
7. Structure your responses clearly with bullet points or sections when appropriate
8. Always include appropriate disclaimers about seeking professional medical care

RESPONSE STRUCTURE:
- Start with a direct answer to the question
- Provide detailed explanation based on available information
- Include relevant medical context
- End with appropriate medical disclaimer

Remember: You are providing information, not diagnosing or prescribing treatment."""
    
    def create_user_prompt(self, question: str, context: str, entities: List[str]) -> str:
        """Create user prompt with question, context, and entities"""
        prompt = f"""Medical Question: {question}

Medical Database Information:
{context}

Identified Medical Entities: {', '.join(entities) if entities else 'None identified'}

Please provide a comprehensive, informative response to this medical question. Use the database information as your primary source, and supplement with general medical knowledge as needed. Ensure your response is accurate, helpful, and includes appropriate medical disclaimers."""
        
        return prompt
    
    def prepare_context(self, retrieval_results: List[Dict]) -> str:
        """Prepare context from retrieval results"""
        if not retrieval_results:
            return "No specific information found in the medical database for this question."
        
        context_parts = []
        for i, result in enumerate(retrieval_results[:3], 1):
            context_part = f"""
Result {i} (Relevance Score: {result.get('score', 0):.2f}):
Topic: {result.get('focus', 'Unknown')}
Question: {result.get('question', 'No question available')}
Answer: {result.get('answer', 'No answer available')}
Category: {result.get('category', 'Unknown')}
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def fallback_response(self, question: str, retrieval_results: List[Dict]) -> str:
        """Provide fallback response when LLM is not available"""
        if not retrieval_results:
            return """I apologize, but I couldn't find specific information about your question in the medical database, and the AI enhancement service is currently unavailable.

For accurate medical information and advice, please:
• Consult with a qualified healthcare professional
• Contact your doctor or medical provider
• Visit reputable medical websites like Mayo Clinic, WebMD, or MedlinePlus
• Call a medical helpline if you have urgent concerns

**Medical Disclaimer**: This chatbot provides general information only and should not replace professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with questions about medical conditions."""
        
        # Use the best matching result from the database
        best_result = retrieval_results[0]
        
        response = f"""Based on the medical database, here's information about your question:

**Topic**: {best_result.get('focus', 'Medical Information')}

**Answer**: {best_result.get('answer', 'No specific answer available')}

**Category**: {best_result.get('category', 'General Medical')}

**Additional Context**:
"""
        
        # Add information from other results if available
        if len(retrieval_results) > 1:
            response += "\nRelated information from the database:\n"
            for result in retrieval_results[1:3]:
                if result.get('answer'):
                    response += f"• {result['answer'][:200]}...\n"
        
        response += """

**Medical Disclaimer**: This information is for educational purposes only and should not replace professional medical advice. Always consult with qualified healthcare providers for personalized medical guidance, diagnosis, and treatment recommendations."""
        
        return response
    
    def analyze_question_intent(self, question: str) -> Dict[str, str]:
        """Analyze the intent and type of medical question"""
        if self.client is None:
            return self.basic_intent_analysis(question)
        
        try:
            prompt = f"""Analyze this medical question and provide the following information in JSON format:

Question: "{question}"

Provide analysis in this JSON structure:
{{
    "intent": "information_seeking|symptom_inquiry|treatment_question|prevention_question|diagnosis_question|general_inquiry",
    "urgency": "low|medium|high",
    "medical_area": "general|cardiology|neurology|oncology|dermatology|etc",
    "question_type": "what|how|why|when|where|should_i",
    "requires_professional": "yes|no|maybe"
}}"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=200,
                temperature=0.1
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error analyzing question intent: {e}")
            return self.basic_intent_analysis(question)
    
    def basic_intent_analysis(self, question: str) -> Dict[str, str]:
        """Basic intent analysis without LLM"""
        question_lower = question.lower()
        
        # Determine intent
        if any(word in question_lower for word in ['what is', 'what are', 'define']):
            intent = "information_seeking"
        elif any(word in question_lower for word in ['symptom', 'pain', 'feel', 'experiencing']):
            intent = "symptom_inquiry"
        elif any(word in question_lower for word in ['treat', 'treatment', 'cure', 'medicine']):
            intent = "treatment_question"
        elif any(word in question_lower for word in ['prevent', 'avoid', 'stop']):
            intent = "prevention_question"
        else:
            intent = "general_inquiry"
        
        # Determine urgency
        if any(word in question_lower for word in ['emergency', 'urgent', 'severe', 'critical']):
            urgency = "high"
        elif any(word in question_lower for word in ['chronic', 'ongoing', 'persistent']):
            urgency = "medium"
        else:
            urgency = "low"
        
        return {
            "intent": intent,
            "urgency": urgency,
            "medical_area": "general",
            "question_type": "general",
            "requires_professional": "maybe"
        }
