"""
Simple Local RAG Assistant - No C++ build tools needed!
Works on Windows, Mac, Linux without any compiler issues
"""
#run using:  python -m streamlit run web_app.py
import os
from pathlib import Path
import numpy as np
from typing import List

# Install: pip install sentence-transformers pypdf ollama numpy

from sentence_transformers import SentenceTransformer
import ollama


class Document:
    """Simple document chunk"""
    def __init__(self, text, source):
        self.text = text
        self.source = source


class SimpleRAG:
    def __init__(self, docs_folder="./documents", model="llama3.2:1b"):
        """
        Simple RAG - no vector database needed!
        
        Args:
            docs_folder: Folder with your documents
            model: Ollama model (llama3.2:1b is fastest)
        """
        self.docs_folder = docs_folder
        self.model = model
        self.chunks = []
        self.embeddings = None
        
        Path(docs_folder).mkdir(exist_ok=True)
        
        print(f"⚡ Simple RAG Assistant ({model})")
        print("Loading embedding model...")
        
        # Small, fast embedding model
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Ready!\n")
    
    def load_documents(self):
        """Load PDFs and text files"""
        
        print(f"Loading documents from {self.docs_folder}...")
        
        doc_folder = Path(self.docs_folder)
        all_texts = []
        
        # Load PDFs
        try:
            from pypdf import PdfReader
            for pdf_path in doc_folder.glob("**/*.pdf"):
                try:
                    reader = PdfReader(str(pdf_path))
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    
                    if text.strip():
                        all_texts.append((text, pdf_path.name))
                        print(f"  ✓ {pdf_path.name}")
                except Exception as e:
                    print(f"  ✗ {pdf_path.name}: {e}")
        except ImportError:
            print("  pypdf not installed - skipping PDFs")
        
        # Load text files
        for txt_path in doc_folder.glob("**/*.txt"):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    if text.strip():
                        all_texts.append((text, txt_path.name))
                        print(f"  ✓ {txt_path.name}")
            except Exception as e:
                print(f"  ✗ {txt_path.name}: {e}")
        
        if not all_texts:
            print("⚠️  No documents found! Add .pdf or .txt files to 'documents' folder.")
            return False
        
        # Split into chunks
        print("\nCreating chunks...")
        self.chunks = []
        for text, source in all_texts:
            chunks = self._split_text(text, chunk_size=500)
            for chunk in chunks:
                self.chunks.append(Document(chunk, source))
        
        print(f"Created {len(self.chunks)} chunks")
        
        # Create embeddings
        print("Creating embeddings...")
        chunk_texts = [c.text for c in self.chunks]
        self.embeddings = self.encoder.encode(chunk_texts, show_progress_bar=True)
        
        print("✓ Documents loaded!\n")
        return True
    
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
        """Find most relevant chunks using cosine similarity"""
        
        if self.embeddings is None:
            return []
        
        # Encode query
        query_embedding = self.encoder.encode([query])[0]
        
        # Calculate cosine similarity
        similarities = np.dot(self.embeddings, query_embedding)
        similarities /= (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding))
        
        # Get top k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [self.chunks[i] for i in top_indices]
    
    def query(self, question: str):
        """Ask a question"""
        
        if not self.chunks:
            print("No documents loaded!")
            return
        
        # Find relevant chunks
        relevant = self._find_relevant(question, top_k=2)
        
        if not relevant:
            print("No relevant information found.")
            return
        
        # Build context
        context = "\n\n".join([doc.text for doc in relevant])
        
        # Create prompt
        prompt = f"""Answer the question based on this context. Be brief and direct.

Context: {context}

Question: {question}

Answer:"""
        
        print(f"\n💡 ", end="", flush=True)
        
        # Get response with streaming
        response = ""
        try:
            for chunk in ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                stream=True,
            ):
                text = chunk['message']['content']
                print(text, end="", flush=True)
                response += text
        except Exception as e:
            print(f"\nError: {e}")
            print("\nMake sure:")
            print("1. Ollama is running (try: ollama serve)")
            print("2. Model is downloaded (try: ollama pull llama3.2:1b)")
            return
        
        print("\n")
        return response
    
    def chat(self):
        """Interactive chat mode"""
        
        print("="*50)
        print("💬 Chat Mode (type 'quit' to exit)")
        print("="*50)
        
        while True:
            try:
                question = input("\n🔍 You: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nBye! 👋")
                    break
                
                if not question:
                    continue
                
                self.query(question)
                
            except KeyboardInterrupt:
                print("\n\nBye! 👋")
                break


def main():
    """Main function"""
    
    print("""
╔══════════════════════════════════════════════════════════╗
║        ⚡ SIMPLE LOCAL RAG - NO BUILD TOOLS NEEDED! ⚡   ║
╚══════════════════════════════════════════════════════════╝

Quick Setup:
1. Install Ollama: https://ollama.ai
2. Pull model: ollama pull llama3.2:1b
3. Install packages: pip install -r requirements.txt
4. Add documents to 'documents' folder
5. Run: python app.py

""")
    
    rag = SimpleRAG(
        docs_folder="./documents",
        model="llama3.2:1b"
    )
    
    if rag.load_documents():
        rag.chat()
    else:
        print("\nAdd some .pdf or .txt files to the 'documents' folder!")


if __name__ == "__main__":
    main()