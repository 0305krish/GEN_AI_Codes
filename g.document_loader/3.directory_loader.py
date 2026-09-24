from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

loader = DirectoryLoader(
    path = "path of the directory",
    glob = "*.pdf",#pattern
    loader_cls = PyPDFLoader
)

# docs = loader.load()

#if we have more number of pdf or pages then instead of load we can use 
#lazy load  as it creates a generator and load the document one by one in the memory

docs = loader.lazy_load()
