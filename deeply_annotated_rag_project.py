# # Proelections RAG System - Enhanced with Chroma and Groq
# This system extracts text from PDFs, embeds document chunks into vector space using HuggingFace E5, stores/retrieves via Chroma, and optionally generates answers using OpenAI.
# 


# # Import Libraries


from langchain.vectorstores import Chroma  # Import necessary library
from langchain_huggingface import HuggingFaceEmbeddings  # Import necessary library
from langchain.schema import Document  # Import necessary library
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Import necessary library
import torch  # Import necessary library
import os  # Import necessary library
import glob  # Import necessary library
from typing import List, Dict, Any  # Import necessary library
import PyPDF2  # Import necessary library
import pdfplumber  # Import necessary library
from pathlib import Path  # Import necessary library
from groq import Groq  # Import necessary library


# # Configuration


GROQ_API_KEY = "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"  # You'll need to get this from https://console.groq.com/  # Your Groq API key (keep this secret!)

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)  # Your Groq API key (keep this secret!)

# Local PDF directory
PDF_DIRECTORY = r"C:\Users\default.LAPTOP-4HP0OBME\OneDrive\Documents\GitHub\RAG\pro_elect\OneDriv"  # Path to your folder containing HOA PDFs

# Chroma database directory
CHROMA_DB_DIR = "./chroma_db"  # Directory where the Chroma vector DB will be saved

# Chunk configs
CHUNK_CONFIGS = {  # Configurable chunk sizes for splitting documents
    "default": {"size": 800, "overlap": 200},
    "small": {"size": 400, "overlap": 100},
    "large": {"size": 1200, "overlap": 300}
}



# # Embedding Model Setup


def setup_embedding_model(silent: bool = False):
    # Sets up the E5 embedding model with normalization and batch processing
    # Uses GPU (cuda) if available, otherwise defaults to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Detects and sets the computation device
    if not silent:
        print(f"🖥️ Using device: {device}")
    try:
        return HuggingFaceEmbeddings(
            model_name="intfloat/e5-base-v2",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32}  # Normalizes embeddings for consistent cosine similarity
        )
    except Exception as e:
        print(f"Error loading E5: {e}")
        return None


# # PDF Text Extraction


def extract_text_from_pdf(pdf_path: str) -> str:
    # Extracts text from PDF using pdfplumber primarily, PyPDF2 as a fallback
    """Extract text from PDF using pdfplumber as primary method, PyPDF2 as fallback."""
    text = ""
    
    try:
        # Try pdfplumber first (better text extraction)
        with pdfplumber.open(pdf_path) as pdf:  # Opens PDF and extracts text per page with better formatting
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"pdfplumber failed for {pdf_path}, trying PyPDF2: {e}")
        
        # Fallback to PyPDF2
        try:
            with open(pdf_path, 'rb') as file:  # Fallback method: PyPDF2 reads PDF as binary
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e2:
            print(f"PyPDF2 also failed for {pdf_path}: {e2}")
            return ""
    
    return text.strip()

def find_and_process_pdfs(directory: str) -> List[Dict[str, str]]:
    # Scans a directory recursively for PDFs and extracts their content
    """Recursively find all PDF files and extract their text."""
    pdf_files = []
    
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return pdf_files
    
    # Recursively find all PDF files
    for pdf_path in glob.glob(os.path.join(directory, "**/*.pdf"), recursive=True):  # Recursively finds all .pdf files in nested folders
        print(f"Processing: {pdf_path}")
        text = extract_text_from_pdf(pdf_path)
        
        if text:
            pdf_files.append({
                "filename": os.path.basename(pdf_path),
                "full_path": pdf_path,
                "text": text
            })
        else:
            print(f"No text extracted from: {pdf_path}")
    
    print(f"Successfully processed {len(pdf_files)} PDF files")
    return pdf_files


# # Document Processing and Chunking


def prepare_documents_enhanced(text: str, source_filename: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    # Splits extracted text into chunks and attaches metadata tags
    """Clean and chunk documents with enhanced metadata."""
    
    if not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n=== ", "\nARTICLE ", "\nSECTION ", "\n\n", "\n", ". ", " "],  # Custom chunking separators (headings, paragraphs, etc.)
        length_function=len,
        is_separator_regex=False,
    )

    raw_docs = splitter.create_documents([text])
    enhanced_docs = []

    for i, doc in enumerate(raw_docs):
        content = doc.page_content.lower()
        metadata = {  # Adds custom tags to each chunk for future filtering or search
            "source_file": source_filename,
            "chunk_id": i,
            "chunk_size": len(doc.page_content),
            "has_voting": any(t in content for t in ["vote", "ballot", "election", "poll", "voting", "electoral"]),
            "has_proxy": "proxy" in content,
            "has_director": any(t in content for t in ["director", "board", "officer", "president", "secretary", "treasurer"]),
            "has_quorum": "quorum" in content,
            "has_notice": any(t in content for t in ["notice", "notification", "notify", "inform"]),
            "has_meeting": any(t in content for t in ["meeting", "assembly", "session", "gathering"]),
        }
        enhanced_docs.append(Document(page_content=doc.page_content, metadata=metadata))

    return enhanced_docs


# # Vector Store Initialization (Chroma)


def initialize_chroma_db(documents: List[Document], embedding_model, persist_directory: str = CHROMA_DB_DIR):  # Directory where the Chroma vector DB will be saved
    """Initialize Chroma vector store with documents."""
    
    if not documents:
        print("No documents to index")
        return None
    
    try:
        # Create Chroma vector store
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=persist_directory  # Where to save the Chroma DB locally
        )
        
        # Persist the database
        vector_store.persist()  # Saves the DB to disk so it doesn’t have to reprocess next time
        print(f"Indexed {len(documents)} document chunks in Chroma DB")
        
        return vector_store
        
    except Exception as e:
        print(f"Error initializing Chroma DB: {e}")
        return None

def load_existing_chroma_db(embedding_model, persist_directory: str = CHROMA_DB_DIR):  # Directory where the Chroma vector DB will be saved
    """Load existing Chroma vector store."""
    
    try:
        vector_store = Chroma(
            persist_directory=persist_directory,  # Where to save the Chroma DB locally
            embedding_function=embedding_model
        )
        return vector_store
    except Exception as e:
        print(f"Error loading existing Chroma DB: {e}")
        return None


# Groq Answer Generation


def generate_answer_with_groq(question: str, retrieved_chunks: List[Document]) -> Dict[str, Any]:
    # Uses Groq (like OpenAI) to generate an answer based on document chunks
    """Generate answer using Groq based on retrieved chunks."""
    
    if not retrieved_chunks:
        return {"error": "No relevant chunks found"}
    
    # Construct context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        source_file = chunk.metadata.get('source_file', 'Unknown')
        context_parts.append(f"[Source {i+1}: {source_file}]\n{chunk.page_content}\n")  # Combines all retrieved chunks into a single context block
    
    context = "\n".join(context_parts)
    
    # Create prompt for Groq
    prompt = f"""You are an expert assistant helping with document analysis. Based on the provided context from various documents, please answer the user's question accurately and comprehensively.  # Constructs the prompt that Groq will use to generate a response

Context from documents:
{context}

User Question: {question}

Please provide a detailed answer based on the context above. If the context doesn't contain enough information to fully answer the question, please mention what information is available and what might be missing. Always cite which source document(s) you're referencing in your answer.

Answer:"""
    
    try:
        # Groq API call (same format as OpenAI but faster and free!)
        response = client.chat.completions.create(
            model="llama3-8b-8192",  # Fast and capable model
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes documents and provides accurate, well-sourced answers."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        
        answer = response.choices[0].message.content.strip()
        
        return {
            "answer": answer,
            "sources": [chunk.metadata.get('source_file', 'Unknown') for chunk in retrieved_chunks],
            "num_chunks_used": len(retrieved_chunks)
        }
        
    except Exception as e:
        return {"error": f"Groq API error: {str(e)}"}


# # Main RAG Query Function


def query_rag_system(question: str, vector_store=None, k: int = 3, use_groq: bool = True) -> Dict[str, Any]:
    # Orchestrates chunk retrieval and optional LLM-based answer generation
    """Main RAG query function that accepts any user question."""
    
    if vector_store is None:
        return {"error": "Vector store not initialized"}
    
    if not question.strip():
        return {"error": "Empty question provided"}
    
    try:
        # Retrieve relevant chunks using Chroma
        results_with_scores = vector_store.similarity_search_with_score(question, k=k)  # Finds top K relevant chunks based on vector similarity
        
        if not results_with_scores:
            return {"error": "No relevant documents found"}
        
        # Extract documents and scores
        retrieved_docs = [doc for doc, score in results_with_scores]
        similarity_scores = [1.0 - score for doc, score in results_with_scores]  # Convert distance to similarity  # Converts distance to similarity (1 = perfect match)
        
        # Generate answer using Groq if requested
        if use_groq:
            ai_response = generate_answer_with_groq(question, retrieved_docs)
            if "error" in ai_response:
                return ai_response
            
            return {
                "question": question,
                "ai_answer": ai_response["answer"],
                "sources": ai_response["sources"],
                "similarity_scores": similarity_scores,
                "retrieved_chunks": [doc.page_content[:200] + "..." for doc in retrieved_docs],
                "num_chunks": len(retrieved_docs)
            }
        else:
            # Return raw chunks without Groq processing
            return {
                "question": question,
                "similarity_scores": similarity_scores,
                "retrieved_chunks": [doc.page_content for doc in retrieved_docs],
                "sources": [doc.metadata.get('source_file', 'Unknown') for doc in retrieved_docs],
                "num_chunks": len(retrieved_docs)
            }
            
    except Exception as e:
        return {"error": f"Query error: {str(e)}"}



# # System Setup Function


def setup_rag_system(chunk_config: str = "default", force_rebuild: bool = False):
    # End-to-end initialization: loads or builds the RAG pipeline
    """Initialize the complete RAG system."""
    
    print("🚀 Initializing RAG System...")
    
    # Setup embedding model
    embedding_model = setup_embedding_model()
    if embedding_model is None:
        return None, None
    
    # Check if we should rebuild or load existing database
    if force_rebuild or not os.path.exists(CHROMA_DB_DIR):  # Directory where the Chroma vector DB will be saved
        print("📚 Processing PDF files...")
        
        # Find and process PDF files
        pdf_data = find_and_process_pdfs(PDF_DIRECTORY)  # Path to your folder containing HOA PDFs
        
        if not pdf_data:
            print("❌ No PDF files found or processed")
            return None, None
        
        # Prepare documents with chunking
        all_documents = []
        chunk_size = CHUNK_CONFIGS[chunk_config]["size"]  # Configurable chunk sizes for splitting documents
        chunk_overlap = CHUNK_CONFIGS[chunk_config]["overlap"]  # Configurable chunk sizes for splitting documents
        
        for pdf_info in pdf_data:
            docs = prepare_documents_enhanced(
                pdf_info["text"], 
                pdf_info["filename"], 
                chunk_size, 
                chunk_overlap
            )
            all_documents.extend(docs)
        
        print(f"📄 Created {len(all_documents)} document chunks")
        
        # Initialize Chroma database
        vector_store = initialize_chroma_db(all_documents, embedding_model)
        
    else:
        print("📁 Loading existing Chroma database...")
        vector_store = load_existing_chroma_db(embedding_model)
    
    if vector_store is None:
        print("❌ Failed to initialize vector store")
        return None, None
    
    print("✅ RAG System ready!")
    return vector_store, embedding_model


# # Interactive Query Function


def ask_question(question: str, detailed: bool = False):
    # Convenience function to ask a question and pretty-print the answer
    """Convenient function to ask questions and get formatted responses."""
    
    if 'vector_store' not in globals() or vector_store is None:
        print("❌ Vector store not initialized. Run setup_rag_system() first.")
        return
    
    print(f"\n🤔 Question: {question}")
    print("-" * 60)
    
    result = query_rag_system(question, vector_store, k=3, use_groq=True)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"🤖 AI Answer:")
    print(result['ai_answer'])
    
    if detailed:
        print(f"\n📊 Details:")
        print(f"  • Sources: {', '.join(set(result['sources']))}")
        print(f"  • Chunks used: {result['num_chunks']}")
        print(f"  • Similarity scores: {[f'{score:.3f}' for score in result['similarity_scores']]}")
        
        print(f"\n📄 Retrieved chunks preview:")
        for i, chunk in enumerate(result['retrieved_chunks']):
            print(f"  Chunk {i+1}: {chunk}")
    
    return result


# # Initialize the System


print("🎯 RAG System Ready to Initialize!")
print("\nTo get started, run:")
print("vector_store, embedding_model = setup_rag_system()")
print("\nThen ask questions with:")
print('ask_question("How are HOA board elections conducted?")')


 vector_store, embedding_model = setup_rag_system(chunk_config="default", force_rebuild=False)


# # Example Questions 


# Ask a question with a clean, formatted response
ask_question("For Highland view How are HOA board elections conducted?")


# Ask a question and see detailed metrics
ask_question("What are the voting requirements for board meetings at Cardigan?", detailed=True)

