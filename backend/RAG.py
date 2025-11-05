import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv

load_dotenv()


chroma_path = "C:\Users\Administrator\Desktop\medpic2_OUT\medpic2\backend\chroma_store"
collection_name = "drug_info"
embedding_model_name = "all-MiniLM-L6-v2"
query_text = "What is the common side effect of aspirin?"
Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)




from llama_index.llms.openai import OpenAI
Settings.llm = OpenAI(model="gpt-4o-mini")

chroma_client = chromadb.PersistentClient(path=chroma_path)
chroma_collection = chroma_client.get_collection(name=collection_name)
print(f"Collection '{collection_name}' has a count of: {chroma_collection.count()}")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
query_engine = index.as_query_engine()


print(f"\nExecuting query: '{query_text}'")
response = query_engine.query(query_text)
print("\n--- Final Response ---")
print(response)