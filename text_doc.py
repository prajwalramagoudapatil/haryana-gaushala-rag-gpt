from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
# from langchain_core.documents import Document
import os
# from rapidfuzz import fuzz

CHROMA_PERSIST_DIR = "chroma_text_db5"
TEXT_FILE_PATH = "extracted_gau_shala_data.txt"
# Text_FILE = 'test_text.txt'
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
add_documents = not os.path.exists(CHROMA_PERSIST_DIR)

# Create the Embeddings object
print(f"Initializing HuggingFace Embeddings with model: {EMBEDDING_MODEL}...")
# embeddings = HuggingFaceEmbeddings(
#     model_name=EMBEDDING_MODEL,
#     model_kwargs={'device': 'cpu'}
# )
emb_model = HuggingFaceEmbeddings(
    model_name="intfloat/e5-base-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

if add_documents:
    loader = TextLoader(TEXT_FILE_PATH , encoding="utf-8")
    documents = loader.load()

    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,       # reliable for tables + names
        chunk_overlap=30,     # gives better context to retriever
        separators=["\n\n", "\n", " ", "", ". "]
    )

    docs = text_splitter.split_documents(documents)

    # Store the embeddings in a local vector store (ChromaDB) 
    print(f"Creating and persisting ChromaDB at {CHROMA_PERSIST_DIR}...")

    # If the directory exists, it will load the existing database.
    # If it doesn't, it will create a new one.
    vectordb = Chroma.from_documents(
        collection_name="gau_shala_col",
        documents=docs, 
        embedding=emb_model, 
        persist_directory=CHROMA_PERSIST_DIR
    )

    print(f"ChromaDB creation complete. Total chunks indexed: {len(docs)}")
else:

    vectordb = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=emb_model,
        collection_name="gau_shala_col"
    )

retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 15
    }
)

# high_quality_retriever = vectordb.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "score_threshold": 0.3, 
#         "k": 7
#     }
# )

if __name__ == "__main__":

    while True:
        question = input('\n ----> Enter query(type "e" to exit): ')

        if question.lower() in ['q', 'e', 'quit', 'exit']:
            break

        result = retriever.invoke(question)

        for doc in result:
            print("\n-Retvr:->> ", doc.page_content)

        # result = high_quality_retriever.invoke(question)
        # print("\nTop similar docs USING [high_quality_retriever] \n  length:", len(result), '\n')
        # for r in result:
        #     print("\n-high qlty:-")
        #     print(r.page_content)
            
        #     try :
        #         print("Metadata:", r['score'])
        #     except :
        #         print("score: ", r)
        #     finally :
        #         pass
            
        # print(result, '\n', (result))
    print("Exiting...")

