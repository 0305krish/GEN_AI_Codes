#used for mostly textual data but not good for image data

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('b.test_pdf')

docs = loader.load()

print(len(docs)) #no of pages in pdf
