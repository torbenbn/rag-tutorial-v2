from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
import os

def get_embedding_function():
    # embeddings = BedrockEmbeddings(credentials_profile_name="default", region_name="us-east-1")
    embeddings = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBEDDINGS_MODEL"), base_url=os.getenv("OLLAMA_BASE_URL"))
    return embeddings
