from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader('../g.document_loader/a.cricket.txt', encoding = 'utf-8')

docs = loader.load()

text = docs[0].page_content

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0, #tells the number of overlap between 2 chunks
    separator = ''
)

result = splitter.split_text(text)

print(result[0])