import streamlit as st
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_response(llm_response: str, retrieval_results: List[Dict], 
                   entities: List[str]) -> Dict[str, any]:
    """Format the final response with source information"""
    
    # Determine if LLM was used
    llm_enhanced = "informational purposes only" in llm_response.lower() or len(llm_response) > 200
    
    # Get best dataset match if available
    best_match = None
    confidence = 0.0
    
    if retrieval_results:
        best_result = retrieval_results[0]
        best_match = f"{best_result.get('focus', 'Unknown topic')} - {best_result.get('question', 'No question')}"
        confidence = best_result.get('score', 0.0)
    
    # Prepare source information
    source_info = {
        "dataset_match": best_match,
        "entities": entities if entities else [],
        "confidence": confidence,
        "llm_enhanced": llm_enhanced,
        "num_sources": len(retrieval_results)
    }
    
    return {
        "response": llm_response,
        "source": source_info
    }

def add_medical_disclaimer():
    """Add medical disclaimer to the UI"""
    with st.expander("⚠️ Important Medical Disclaimer - Please Read", expanded=False):
        st.warning("""
        **MEDICAL DISCLAIMER**
        
        This chatbot is for **informational and educational purposes only** and should not be used as a substitute for professional medical advice, diagnosis, or treatment.
        
        **Important Guidelines:**
        - Always seek the advice of qualified healthcare providers
        - Never disregard professional medical advice based on information from this chatbot
        - If you have a medical emergency, call emergency services immediately
        - This tool is not intended to diagnose, treat, cure, or prevent any disease
        - Information provided may not be current or complete
        
        **When to Seek Immediate Medical Attention:**
        - Chest pain or difficulty breathing
        - Severe injuries or bleeding
        - Loss of consciousness
        - Severe allergic reactions
        - Any life-threatening emergency
        
        By using this chatbot, you acknowledge that you understand these limitations and agree to use the information responsibly.
        """)

def display_search_results(results: List[Dict], max_results: int = 3):
    """Display search results in a formatted way"""
    if not results:
        st.info("No relevant results found in the medical database.")
        return
    
    st.subheader(f"📚 Found {len(results)} relevant results:")
    
    for i, result in enumerate(results[:max_results], 1):
        with st.expander(f"Result {i}: {result.get('focus', 'Medical Information')} (Score: {result.get('score', 0):.2f})"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write("**Question:**", result.get('question', 'No question available'))
                st.write("**Answer:**", result.get('answer', 'No answer available'))
                
                if result.get('synonym'):
                    st.write("**Related Terms:**", result.get('synonym'))
            
            with col2:
                st.write("**Category:**", result.get('category', 'Unknown'))
                st.write("**Type:**", result.get('question_type', 'General'))
                
                if result.get('semantic_type'):
                    st.write("**Semantic Type:**", result.get('semantic_type'))

def create_entity_badges(entities: List[str]):
    """Create styled badges for medical entities"""
    if not entities:
        return ""
    
    # Create HTML badges
    badges_html = ""
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57", "#FF9FF3"]
    
    for i, entity in enumerate(entities[:10]):  # Limit to 10 entities
        color = colors[i % len(colors)]
        badges_html += f"""
        <span style="
            background-color: {color};
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 12px;
            margin: 2px;
            display: inline-block;
        ">{entity}</span>
        """
    
    return badges_html

def validate_question(question: str) -> tuple[bool, str]:
    """Validate user question"""
    if not question or not question.strip():
        return False, "Please enter a question."
    
    if len(question.strip()) < 5:
        return False, "Please enter a more detailed question."
    
    if len(question) > 500:
        return False, "Question is too long. Please keep it under 500 characters."
    
    # Check for potentially harmful content
    harmful_keywords = ['suicide', 'self-harm', 'overdose', 'kill myself']
    if any(keyword in question.lower() for keyword in harmful_keywords):
        return False, "If you're experiencing a mental health crisis, please contact emergency services or a crisis helpline immediately."
    
    return True, ""

def get_question_suggestions() -> List[str]:
    """Get sample questions to help users"""
    return [
        "What is diabetes and how is it treated?",
        "What are the symptoms of high blood pressure?",
        "How can I prevent heart disease?",
        "What causes migraine headaches?",
        "How is pneumonia diagnosed and treated?",
        "What are the risk factors for stroke?",
        "How can I manage arthritis pain?",
        "What is the difference between Type 1 and Type 2 diabetes?",
        "What are the early signs of cancer?",
        "How can I improve my immune system?"
    ]

def log_user_interaction(question: str, response_quality: str = "unknown"):
    """Log user interactions for analysis"""
    try:
        logger.info(f"User question: {question[:100]}... | Response quality: {response_quality}")
    except Exception as e:
        logger.error(f"Error logging interaction: {e}")

def check_system_health() -> Dict[str, bool]:
    """Check if all system components are working"""
    health_status = {
        "data_loaded": False,
        "retrieval_ready": False,
        "entities_ready": False,
        "llm_ready": False
    }
    
    try:
        # Check session state components
        if hasattr(st.session_state, 'medquad_processor') and st.session_state.medquad_processor is not None:
            health_status["data_loaded"] = True
        
        if hasattr(st.session_state, 'retriever') and st.session_state.retriever is not None:
            health_status["retrieval_ready"] = True
        
        if hasattr(st.session_state, 'entity_recognizer') and st.session_state.entity_recognizer is not None:
            health_status["entities_ready"] = True
        
        if hasattr(st.session_state, 'llm') and st.session_state.llm is not None:
            health_status["llm_ready"] = True
    
    except Exception as e:
        logger.error(f"Error checking system health: {e}")
    
    return health_status

def create_download_chat_history():
    """Create downloadable chat history"""
    if not st.session_state.messages:
        return None
    
    try:
        chat_text = "Medical Q&A Chat History\n"
        chat_text += "=" * 50 + "\n\n"
        
        for i, message in enumerate(st.session_state.messages, 1):
            if message["role"] == "user":
                chat_text += f"Question {i//2 + 1}: {message['content']}\n\n"
            else:
                response_content = message["content"]
                if isinstance(response_content, dict):
                    chat_text += f"Answer: {response_content.get('response', 'No response')}\n\n"
                else:
                    chat_text += f"Answer: {response_content}\n\n"
                chat_text += "-" * 30 + "\n\n"
        
        chat_text += "\nGenerated by Medical Q&A Chatbot\n"
        chat_text += "Disclaimer: This information is for educational purposes only.\n"
        
        return chat_text
    
    except Exception as e:
        logger.error(f"Error creating chat history: {e}")
        return None

def display_system_metrics():
    """Display system performance metrics"""
    try:
        if hasattr(st.session_state, 'medquad_processor') and st.session_state.medquad_processor:
            stats = st.session_state.medquad_processor.get_statistics()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Q&A Pairs", stats.get('total_questions', 0))
            
            with col2:
                st.metric("Disease Categories", stats.get('disease_count', 0))
            
            with col3:
                st.metric("Chat Messages", len(st.session_state.messages))
            
            with col4:
                health = check_system_health()
                ready_components = sum(health.values())
                st.metric("System Components", f"{ready_components}/4")
    
    except Exception as e:
        logger.error(f"Error displaying metrics: {e}")
