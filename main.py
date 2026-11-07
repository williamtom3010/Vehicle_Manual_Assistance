import torch    
from transformers import AutoTokenizer, AutoModelForCausalLM   
from marker.converters.pdf import PdfConverter    
from marker.models import create_model_dict   
from marker.output import text_from_rendered   
from marker.config.parser import ConfigParser   
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import os
import models


# get data
DATA_DIR   = "data/YTB-Fronx_99011M74T04-74W.pdf"
INDEX_DIR  = "fronx_manual_vector_db"
os.makedirs(INDEX_DIR, exist_ok=True)

# extract data


# converter = PdfConverter(
#     artifact_dict=create_model_dict(),
# )
# rendered = converter("DATA_DIR")
# text, _, images = text_from_rendered(rendered)


config = {
    "output_format": "markdown",
    # "ADDITIONAL_KEY": "VALUE"
}
config_parser = ConfigParser(config)

converter = PdfConverter(
    config=config_parser.generate_config_dict(),
    artifact_dict=create_model_dict(),
    processor_list=config_parser.get_processors(),
    renderer=config_parser.get_renderer(),
    llm_service=config_parser.get_llm_service()
)

# Will this extract only one page from the pdf or will it extract all pages from pdf and store as .txt ?
rendered = converter(DATA_DIR)
text, _, images = text_from_rendered(rendered)

# save the text data to folder
os.makedirs("data", exist_ok=True)
extracted_text_data_path = "data/text_data.md"
with open(extracted_text_data_path, "w", encoding="utf-8") as f:
    f.write(text)

# load the the .md file

loader = UnstructuredMarkdownLoader(extracted_text_data_path, mode='elements')
data_md = loader.load()

print(f"Loaded elements: {len(data_md)}")
print(data_md[1].page_content)
print(data_md[1].metadata)


# chunk data

loader = TextLoader(extracted_text_data_path, encoding="utf-8")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(documents)
print("Document Chunks:", len(chunks))

# choose embedding model
embedding_model = models.load_embedding_model()


# Vector Database
client = chromadb.PersistentClient(path=INDEX_DIR)
collection = client.get_or_create_collection(
    name="fronx_manual",
    metadata={"hnsw:space": "cosine"},
)

texts = []
metadata = []
for chunk in chunks:
    texts.append(chunk.page_content)
    metadata.append(chunk.metadata)

ids = [f"chunk_{i}" for i in range(len(chunks))]
# store in vector DB
   
document_embeddings = embedding_model.encode(
                                   texts,
                                   batch_size=16,
                                   normalize_embeddings=True,
                                   show_progress_bar=True
                                   )
collection.upsert(
    ids=ids,
    documents=texts,
    embeddings=document_embeddings.tolist(),
    metadatas=metadata,
)

print("Stored:", collection.count())


# ask query by user
query = "How to switch of the automatic engine start stop feature  ?"

# query embed


q_emb = embedding_model.encode([query], prompt_name="query", normalize_embeddings=True)

n_chunks = 4
res = collection.query(
    query_embeddings=q_emb.tolist(), 
    n_results=n_chunks, 
    include=["documents", "metadatas", "distances"]
    )

docs = res["documents"][0]
metas = res["metadatas"][0]
dists = res["distances"][0]

context = "\n\n".join(
    f"[{i+1}] (Section: {m.get('h1','')} > {m.get('h2','')})\n{d}"
    for i, (d, m) in enumerate(zip(docs, metas))
)
# Compute the (cosine) similarity between the query and document embeddings
# retrieve information related to query
similarity = embedding_model.similarity(q_emb, document_embeddings)
print("Similarity : ",similarity)


print("No of Retrieved Chunks : ",len(res["documents"][0]))   # 4  (four chunks)
print("Retrieved Chunks are \n\n : ",docs)



# prompt the generation model

answer = models.gen_ans(query,context)
print(answer)