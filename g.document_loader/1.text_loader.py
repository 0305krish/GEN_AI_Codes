from langchain_community.document_loaders import TextLoader

loader = TextLoader('a.cricket.txt', encoding = 'utf-8')

docs = loader.load()
print(docs) #list
print(docs[0]) #document containing page_content and metadata