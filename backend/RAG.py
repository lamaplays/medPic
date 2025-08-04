""" 
this code is supposed to work, but it doesnt bc print(collection.count()) is 0 bc my chroma code is not compatible with llama index. 
also consider switching to llama workflows since everything is fucking deprecated beside that
cheers


"""



import chromadb
from llama_index.core import VectorStoreIndex 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings


#sfda_brands = load_sfda()


embedding_model = HuggingFaceEmbedding('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="chroma_store")
collection = client.get_or_create_collection("drug_info")
print(collection.count())  

   
# Wrap it with LlamaIndex
# chroma_store = ChromaVectorStore(chroma_collection=collection)
# index = VectorStoreIndex.from_vector_store(
#     vector_store=chroma_store,
#     embed_model=embedding_model,
#      )

# Settings.llm = OpenAI(model="gpt-4o-mini")



# query_engine = index.as_query_engine()
# response = query_engine.query("what are the side effects of Adol?")
# print(response)

# # output is "Empty Response"