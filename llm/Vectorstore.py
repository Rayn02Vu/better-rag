import os

from langchain.document_loaders import TextLoader, PDFMinerLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain.vectorstores import FAISS
from Embedding import embedding_model


INDEXS = {
    "VN-History": ["data/LichsuDang.pdf"],
}


def setup_vectorstore():
    """Setup vectorstore for the given index."""
    for index_name, file_paths in INDEXS.items():
        print(f"Processing index: {index_name}")
        documents = []
        for file_path in file_paths:
            if file_path.endswith('.pdf'):
                loader = PDFMinerLoader(file_path)
            else:
                loader = TextLoader(file_path)
            documents.extend(loader.load())
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        
        vectorstore = FAISS.from_documents(docs, embedding_model)
        vectorstore.save_local(f"vectorstores/{index_name}.faiss")
        print(f"Vectorstore for {index_name} created and saved.")


if __name__ == "__main__":
    if not os.path.exists("vectorstores"):
        os.makedirs("vectorstores")
    setup_vectorstore()
    print("All vectorstores have been set up.")