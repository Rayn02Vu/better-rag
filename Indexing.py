import os
import streamlit as st

from langchain_community.document_loaders import TextLoader, PDFMinerLoader
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from llm.utils import clean_text

@st.cache_resource
def setup_resource():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embedding_model

embedding_model = setup_resource()

INDEXS = {
    "VN-History": ["data/LichsuDang.pdf"],
    "Novel": ["data/TestData.txt"]
}

def setup_vectorstore(index: str, chunk_size: int = 2500, chunk_overlap: int = 100):
    """Setup vectorstore for the given index."""
    if index not in INDEXS:
        return
    print(f"Processing index: {index}")
    file_paths = INDEXS[index]
    documents = []
    for file_path in file_paths:
        if file_path.endswith('.pdf'):
            loader = PDFMinerLoader(file_path)
        else:
            loader = TextLoader(file_path)
        documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n\n", "\n\n", "\n", ". ", ".", " ", ""],
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
    )
    docs = text_splitter.split_documents(documents)
    
    vectorstore = FAISS.from_documents(docs, embedding_model)
    vectorstore.save_local(f"vectorstores/{index}.faiss")
    print(f"Vectorstore for {index} created and saved.")



def setup_advanced_vectorstore(index: str, chunk_size: int = 2500, chunk_overlap: int = 100):
    if index not in INDEXS:
        return
    print(f"Processing index: {index}")
    file_paths = INDEXS[index]

    processed_chunks: list[Document] = []
    current_chapter_title = "Chương không xác định"
    current_section_title = "Mục không xác định"
    current_chapter_num = "N/A" 
    seen_titles = set() 

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  
        chunk_overlap=chunk_overlap, 
        separators=["\n\n", "\n", " ", ""] 
    )

    for file_path in file_paths:
        loader = UnstructuredLoader(file_path, mode="elements")
        elements = loader.load()

        st.write(f"Processing file: {file_path}")
        st.write(f"Number of elements: {len(elements)}")

        for i, element in enumerate(elements):
            page_number = element.metadata.get('page_number', 'N/A')
            element_type = element.metadata.get('category', 'NarrativeText') 
            element_text = clean_text(element.page_content)
            if element_type == 'Title' and element_text and element_text not in seen_titles:
                import re
                chapter_match = re.match(r"^(?:CHƯƠNG|PHẦN|PART)\s+([IVXLCDM\d]+)[:\.]?\s*(.*)$", element_text, re.IGNORECASE)
                
                if chapter_match:
                    current_chapter_num = chapter_match.group(1).strip()
                    current_chapter_title = (chapter_match.group(2) or element_text).strip() 
                    current_section_title = "Mở đầu chương"
                    st.write(f"Detected Chapter: **{element_text}** (Page: {page_number})")

                elif len(element_text.split()) < 10 and len(element_text) > 5: 
                    current_section_title = element_text 
                    st.write(f"Detected Section: {element_text} (Page: {page_number})")
                
                seen_titles.add(element_text) 

            if element_type == 'Title' and element_text in seen_titles:
                continue
                
            if not element_text: 
                continue
            
            chunk_metadata = {
                "source": element.metadata.get("source", os.path.basename(file_path)),
                "page_number": page_number,
                "chapter_title": current_chapter_title,
                "chapter_num": current_chapter_num,
                "section_title": current_section_title,
                "element_type": element_type, 
            }
            chunk = Document(page_content=element_text, metadata=chunk_metadata)
            sub_chunks = text_splitter.split_documents([chunk])
            processed_chunks.extend(sub_chunks)

    vectorstore = FAISS.from_documents(processed_chunks, embedding_model)
    vectorstore.save_local(f"vectorstores/{index}.faiss")
    print(f"Vectorstore for {index} created and saved.")



def get_vectorstore(index: str):
    if index not in INDEXS:
        return None
    vectorstore = FAISS.load_local(
        folder_path=f"vectorstores/{index}.faiss",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    return vectorstore


if __name__ == "__main__":
    setup_advanced_vectorstore("VN-History", chunk_size=2000, chunk_overlap=200)



