import streamlit as st
import pandas as pd
import os
from datetime import datetime
import json

from data_processor import MedQuADProcessor
from retrieval_system import MedicalRetriever
from medical_entities import MedicalEntityRecognizer
from llm_integration import MedicalLLM
from utils import format_response, add_medical_disclaimer

# Page configuration
st.set_page_config(
    page_title="Medical Q&A Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "medquad_processor" not in st.session_state:
    st.session_state.medquad_processor = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "entity_recognizer" not in st.session_state:
    st.session_state.entity_recognizer = None
if "llm" not in st.session_state:
    st.session_state.llm = None

@st.cache_resource
def load_medquad_data():
    """Load and process MedQuAD dataset"""
    try:
        import os
        # Use absolute path
        csv_path = os.path.join(os.getcwd(), "attached_assets", "medquad_dataset_1757220899495.csv")
        processor = MedQuADProcessor(csv_path)
        return processor
    except Exception as e:
        st.error(f"Error loading MedQuAD dataset: {str(e)}")
        return None

@st.cache_resource
def initialize_retriever(_processor):
    """Initialize the medical retrieval system"""
    if _processor is None:
        return None
    try:
        retriever = MedicalRetriever(_processor.get_processed_data())
        return retriever
    except Exception as e:
        st.error(f"Error initializing retriever: {str(e)}")
        return None

@st.cache_resource
def initialize_entity_recognizer():
    """Initialize medical entity recognizer"""
    try:
        recognizer = MedicalEntityRecognizer()
        return recognizer
    except Exception as e:
        st.error(f"Error initializing entity recognizer: {str(e)}")
        return None

@st.cache_resource
def initialize_llm():
    """Initialize LLM integration"""
    try:
        llm = MedicalLLM()
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

def main():
    st.title("🏥 Medical Q&A Chatbot")
    st.markdown("Ask medical questions and get answers from the MedQuAD dataset enhanced with AI.")
    
    # Medical disclaimer
    add_medical_disclaimer()
    
    # Sidebar for settings and information
    with st.sidebar:
        st.header("ℹ️ Information")
        st.markdown("""
        This chatbot uses:
        - **MedQuAD Dataset**: Medical Q&A pairs
        - **Retrieval System**: Finds relevant answers
        - **Entity Recognition**: Identifies medical terms
        - **AI Enhancement**: Provides contextual responses
        """)
        
        # Load data status
        st.header("📊 System Status")
        
        # Initialize components
        if st.session_state.medquad_processor is None:
            with st.spinner("Loading MedQuAD dataset..."):
                st.session_state.medquad_processor = load_medquad_data()
        
        if st.session_state.retriever is None and st.session_state.medquad_processor is not None:
            with st.spinner("Initializing retrieval system..."):
                st.session_state.retriever = initialize_retriever(st.session_state.medquad_processor)
        
        if st.session_state.entity_recognizer is None:
            with st.spinner("Loading medical entity recognizer..."):
                st.session_state.entity_recognizer = initialize_entity_recognizer()
        
        if st.session_state.llm is None:
            with st.spinner("Initializing AI system..."):
                st.session_state.llm = initialize_llm()
        
        # Display status
        components_status = {
            "MedQuAD Dataset": st.session_state.medquad_processor is not None,
            "Retrieval System": st.session_state.retriever is not None,
            "Entity Recognition": st.session_state.entity_recognizer is not None,
            "AI Integration": st.session_state.llm is not None
        }
        
        for component, status in components_status.items():
            if status:
                st.success(f"✅ {component}")
            else:
                st.error(f"❌ {component}")
        
        # Dataset statistics
        if st.session_state.medquad_processor is not None:
            stats = st.session_state.medquad_processor.get_statistics()
            st.header("📈 Dataset Statistics")
            st.metric("Total Questions", stats['total_questions'])
            st.metric("Disease Categories", stats['disease_count'])
            st.metric("Other Categories", stats['other_count'])
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    st.header("💬 Chat Interface")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(message["content"]["response"])
                
                # Show source information if available
                if "source" in message["content"] and message["content"]["source"]:
                    with st.expander("📋 Source Information"):
                        source_info = message["content"]["source"]
                        if "dataset_match" in source_info:
                            st.write("**Dataset Match:**", source_info["dataset_match"])
                        if "entities" in source_info:
                            st.write("**Medical Entities:**", ", ".join(source_info["entities"]))
                        if "confidence" in source_info:
                            st.write("**Confidence Score:**", f"{source_info['confidence']:.2f}")
            else:
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a medical question..."):
        # Check if all components are loaded
        if not all(components_status.values()):
            st.error("⚠️ Some system components are not loaded. Please wait for initialization to complete.")
            return
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching medical database and generating response..."):
                try:
                    # Extract medical entities
                    entities = st.session_state.entity_recognizer.extract_entities(prompt)
                    
                    # Retrieve relevant information from dataset
                    retrieval_results = st.session_state.retriever.search(prompt, top_k=3)
                    
                    # Generate enhanced response with LLM
                    llm_response = st.session_state.llm.generate_response(
                        question=prompt,
                        retrieval_results=retrieval_results,
                        entities=entities
                    )
                    
                    # Format the final response
                    formatted_response = format_response(
                        llm_response=llm_response,
                        retrieval_results=retrieval_results,
                        entities=entities
                    )
                    
                    # Display response
                    st.markdown(formatted_response["response"])
                    
                    # Show source information
                    if formatted_response["source"]:
                        with st.expander("📋 Source Information"):
                            source_info = formatted_response["source"]
                            if "dataset_match" in source_info and source_info["dataset_match"]:
                                st.write("**Best Dataset Match:**", source_info["dataset_match"])
                            if "entities" in source_info and source_info["entities"]:
                                st.write("**Medical Entities Found:**", ", ".join(source_info["entities"]))
                            if "confidence" in source_info:
                                st.write("**Retrieval Confidence:**", f"{source_info['confidence']:.2f}")
                            if "llm_enhanced" in source_info:
                                st.write("**AI Enhanced Response:**", "Yes" if source_info["llm_enhanced"] else "No")
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": formatted_response
                    })
                    
                except Exception as e:
                    error_message = f"❌ Error generating response: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": {"response": error_message, "source": None}
                    })

if __name__ == "__main__":
    main()
