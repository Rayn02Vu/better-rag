from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if __name__ == "__main__":
    str_query = "Đây là một câu test..."
    embed_query = embedding_model.embed_query(str_query)
    print(f"Query: {str_query}")
    print(f"Length: {len(embed_query)}")
    print(f"Vector: {embed_query[:5]}")