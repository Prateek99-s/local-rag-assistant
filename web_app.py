"""
Web-based RAG Assistant - Upload PDFs and ask questions!
Simple web interface - no command line needed
"""

import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
from pypdf import PdfReader
from typing import List
import io

# Page config
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📄",
    layout="wide"
)


class Document:
    """Simple document chunk"""
    def __init__(self, text, source):
        self.text = text
        self.source = source


class WebRAG:
    def __init__(self, model="llama3.2:1b"):
        self.model = model
        self.chunks = []
        self.embeddings = None
        
        # Load encoder only once
        if 'encoder' not in st.session_state:
            with st.spinner("Loading AI model..."):
                st.session_state.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.encoder = st.session_state.encoder
    
    def process_pdf(self, uploaded_file):
        """Process uploaded PDF"""
        try:
            # Read PDF
            pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
            
            # Extract text
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            if not text.strip():
                return False, "Could not extract text from PDF"
            
            # Split into chunks
            chunks = self._split_text(text, chunk_size=500)
            self.chunks = [Document(chunk, uploaded_file.name) for chunk in chunks]
            
            # Create embeddings
            chunk_texts = [c.text for c in self.chunks]
            self.embeddings = self.encoder.encode(chunk_texts, show_progress_bar=False)
            
            return True, f"Processed {len(self.chunks)} chunks from {uploaded_file.name}"
            
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def _split_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Simple text splitter"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1
            
            if current_size >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _find_relevant(self, query: str, top_k: int = 2):
        """Find most relevant chunks"""
        if self.embeddings is None:
            return []
        
        query_embedding = self.encoder.encode([query])[0]
        
        similarities = np.dot(self.embeddings, query_embedding)
        similarities /= (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding))
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [self.chunks[i] for i in top_indices]
    
    def query(self, question: str):
        """Ask a question"""
        if not self.chunks:
            return "Please upload a PDF first!"
        
        # Find relevant chunks
        relevant = self._find_relevant(question, top_k=2)
        
        if not relevant:
            return "No relevant information found."
        
        # Build context
        context = "\n\n".join([doc.text for doc in relevant])
        
        # Create prompt
        prompt = f"""Answer the question based on this context. Be brief and direct.

Context: {context}

Question: {question}

Answer:"""
        
        # Get response
        try:
            response = ""
            for chunk in ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                stream=True,
            ):
                response += chunk['message']['content']
            
            return response
            
        except Exception as e:
            return f"Error: {str(e)}\n\nMake sure Ollama is running and llama3.2:1b is installed!"


def main():
    # Title
    st.title("📄 PDF Chat Assistant")
    st.markdown("Upload a PDF and ask questions about it!")
    
    # Initialize session state
    if 'rag' not in st.session_state:
        st.session_state.rag = WebRAG()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.header("📎 Upload PDF")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF document to chat with"
        )
        
        if uploaded_file is not None:
            if st.button("Process PDF", type="primary"):
                with st.spinner("Processing PDF..."):
                    success, message = st.session_state.rag.process_pdf(uploaded_file)
                    
                    if success:
                        st.success(message)
                        st.session_state.pdf_processed = True
                        st.session_state.messages = []  # Clear chat history
                    else:
                        st.error(message)
        
        st.divider()
        
        st.markdown("### ℹ️ Setup")
        st.markdown("""
        **Requirements:**
        1. Install Ollama from [ollama.ai](https://ollama.ai)
        2. Run: `ollama pull llama3.2:1b`
        3. Run: `pip install -r requirements.txt`
        4. Start app: `streamlit run web_app.py`
        """)
        
        if st.session_state.pdf_processed:
            st.success("✅ PDF loaded and ready!")
        else:
            st.info("👆 Upload a PDF to get started")
    
    # Main chat area
    if not st.session_state.pdf_processed:
        st.info("👈 Please upload a PDF from the sidebar to start chatting!")
    else:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your PDF..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.rag.query(prompt)
                st.markdown(response)
            
            # Add assistant message
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
