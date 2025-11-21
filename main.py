# from langchain_ollama.llms import OllamaLLM

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_groq import ChatGroq

# from vectordb import high_quality_retriever
from text_doc import retriever
import json


with open("config.json") as f:
    config = json.load(f)

api_key = config["groq_api_key"]


print("Raama")


# llm = OllamaLLM(model="llama3.2:1b", temperature=0.1)

llm = ChatGroq(
    groq_api_key=api_key,
    model="llama-3.1-8b-instant",
    temperature=0.2
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert Q&A system. Answer the user's question ONLY based on the following context. "
        "If total count is mensioned in the following context, then that itself is final total count. DO NOT try to calculate or infer anything extra."
        "You are a strict extraction system. Follow these rules EXACTLY:"
        '''
        RULES:
        1. If the context contains a "Total", or any total-like field,
            you MUST use that number exactly as written.
        2. DO NOT add, sum, aggregate, merge, calculate, estimate, or infer totals.
        3. DO NOT use any other numbers in the context to compute totals.
        4. If the user asks for TOTAL, respond:
            "The total is already provided in the context: <value>."
        '''
        "Context: {context}"
    ),
    ("human", "{input}"),
])

document_chain = create_stuff_documents_chain(
    llm, 
    prompt
)

qa_chain = create_retrieval_chain(
    retriever, 
    document_chain
)


while True:
    print('--------------------------------- \n')

    query = input("Ask a question (type 'exit' to quit): ")

    if query.lower() in ["exit", "quit", 'e', 'q']:
        print("Goodbye!")
        break

    result = qa_chain.invoke({"input": query})

    # for k,v in result.items():
    #     print("---\n")
    #     print(k, ': ', v)

    for doc in result['context']:
        print("\n-->>  Related doc:", doc.page_content)

    print("\n--->>> Responce: ", result['answer'])
    # print("\nRelated lines from speech:", result['source_documents'])
    # print("-" * 50)

