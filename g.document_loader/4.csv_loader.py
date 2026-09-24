from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path = "")

docs = loader.load() #for every row we get an object
