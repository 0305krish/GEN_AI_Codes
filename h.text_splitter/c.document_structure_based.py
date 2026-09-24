#when the document is structured on different basis like
#codes, markdown etc based on keywords like class, def etc

from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.PYTHON, #similarly for other languages also
    chunk_size = 300,
    chunk_overlap = 0
)

chunks = splitter.split_text("")

print(chunks[0])