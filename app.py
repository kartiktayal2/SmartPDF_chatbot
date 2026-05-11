# =========================
# IMPORT REQUIRED LIBRARIES
# =========================

# Load PDF files
from langchain_community.document_loaders import PyPDFLoader

# Split large text into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Convert text into embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Vector database
from langchain_chroma import Chroma

# Gemini API (NEW SDK)
from google import genai

# Load environment variables
from dotenv import load_dotenv

# Access environment variables
import os


# =========================
# LOAD API KEY
# =========================

# Load .env file
load_dotenv()

# Get API key
google_api_key = os.getenv("GOOGLE_API_KEY")

# Create Gemini client
client = genai.Client(api_key=google_api_key)

print("\n✅ Gemini Client Initialized!")


# =========================
# STEP 1: LOAD PDF
# =========================

# PDF path
pdf_path = r"C:/Users/karti/OneDrive/Desktop/projects/rag-chatbot/Sample.pdf"

# Load PDF
loader = PyPDFLoader(pdf_path)

# Read PDF
documents = loader.load()

print("✅ PDF Loaded Successfully!")


# =========================
# STEP 2: CHUNKING
# =========================

# Create text splitter
splitter = RecursiveCharacterTextSplitter(

    # Max size of each chunk
    chunk_size=500,

    # Overlapping text between chunks
    chunk_overlap=50
)

# Split into chunks
docs = splitter.split_documents(documents)

print(f"✅ Total Chunks Created: {len(docs)}")


# =========================
# STEP 3: CREATE EMBEDDINGS
# =========================

# Load embedding model
embedding = HuggingFaceEmbeddings(

    # Embedding model
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("✅ Embedding Model Loaded!")


# =========================
# STEP 4: CREATE VECTOR DB
# =========================

# Store embeddings in ChromaDB
vectorstore = Chroma.from_documents(

    # Document chunks
    documents=docs,

    # Embedding model
    embedding=embedding,

    # Save database locally
    persist_directory="./chroma_db"
)

print("✅ Vector Database Created!")


# =========================
# CHATBOT LOOP
# =========================

print("\n======================================")
print("🤖 RAG PDF CHATBOT IS READY!")
print("Type 'exit' to stop the chatbot.")
print("======================================\n")


# Infinite loop for continuous chat
while True:

    # =========================
    # TAKE USER QUESTION
    # =========================

    query = input("\n💬 Ask Your Question: ")

    # Exit condition
    if query.lower() == "exit":
        print("\n👋 Exiting Chatbot...")
        break

    # Ignore empty questions
    if query.strip() == "":
        print("⚠️ Please enter a valid question.")
        continue


    # =========================
    # RETRIEVE RELEVANT CHUNKS
    # =========================

    results = vectorstore.similarity_search(

        # User question
        query,

        # Number of chunks to retrieve
        k=3
    )

    print("\n✅ Relevant Chunks Retrieved!")


    # =========================
    # CREATE CONTEXT
    # =========================

    # Combine retrieved chunks
    context = "\n".join([doc.page_content for doc in results])


    # =========================
    # OPTIONAL DEBUG CONTEXT
    # =========================

    print("\n========== RETRIEVED CONTEXT ==========\n")
    print(context)


    # =========================
    # CREATE FINAL PROMPT
    # =========================

    prompt = f"""
    You are a helpful AI assistant.

    Answer the question using ONLY the context below.

    If the answer is not present in the context,
    say:
    "The answer is not available in the provided PDF."

    Context:
    {context}

    Question:
    {query}
    """


    # =========================
    # SEND TO GEMINI
    # =========================

    try:

        # Generate response
        response = client.models.generate_content(

            # Gemini model
            model="gemini-2.5-flash",

            # Prompt
            contents=prompt
        )


        # =========================
        # PRINT FINAL ANSWER
        # =========================

        print("\n======================================")
        print("📌 USER QUESTION:")
        print("======================================")
        print(query)

        print("\n======================================")
        print("🤖 CHATBOT ANSWER:")
        print("======================================")
        print(response.text)

        print("\n======================================")



    # =========================
    # ERROR HANDLING
    # =========================

    except Exception as e:

        print("\n❌ ERROR OCCURRED:")
        print(e)





