from langchain_community.document_loaders import WebBaseLoader 

url = 'some url of website'

loader = WebBaseLoader(url) #can pass list of url also

docs = loader.load()