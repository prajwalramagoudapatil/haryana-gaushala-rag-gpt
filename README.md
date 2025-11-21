# Haryana Gaushala RAG Search System

This project is a Retrieval-Augmented Generation (RAG) application designed to query district-wise gaushala information from a dataset extracted from official Haryana PDF documents. The system retrieves the most relevant document chunks and answers user queries without hallucinating totals or inferring extra values.

---

## 🚀 Features

- Extract tables from PDF files using `pdfplumber`.
- Store table in Pandas Data frames.
- Add extra column for ditrict name.
- Convert Datafram rows into clean, structured text.
- Split large files using LangChain **Recursive Text Splitter**.
- Embed the text using **intfloat/e5-base-v2** embeddings.
- Store embeddings using **ChromaDB** with persistence.
- Query using **Groq Llama-3.1-8B-Instant** with custom prompts.

---

## 📦 Tech Stack

### **Embeddings**
- Model: `intfloat/e5-base-v2`
- Library: HuggingFaceEmbeddings

### **Vector Database**
- **ChromaDB**
- Persistent directory: `./chroma_store`

### **Language Model**
- **ChatGroq**
- Model: `llama-3.1-8b-instant`
- Temperature: `0.2`

### **LangChain Components**
- RecursiveCharacterTextSplitter
- create_stuff_documents_chain()
- create_retrieval_chain()
- Chroma.from_documents()
- ChatPromptTemplate

---

## 🛠 Folder Structure

```
project/
├── gaushala.pdf
├── extracted_text.txt
│── chroma_textdb/
├── read_pdf.py
├── text_doc.py
├── rag_pipeline.py
├── prompts.py
├── readme.md
└── main.py
```

---

## 📄 PDF → Text Extraction

`pdfplumber` is used to read district-wise tables:

```python
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        tables = page.extract_tables()
        for table in tables:
            df = pd.DataFrame(table[1:], columns=table[0])
            text += df.to_markdown(index=False)
```

The output is saved as `extracted_text.txt`.

---

## ✂️ Text Splitting

Because the PDF contains long tables, we use a Recursive Text Splitter:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=40
)
docs = text_splitter.create_documents([text])
```

---

## 🔤 Embeddings

We use the robust E5 embedding model:

```python
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/e5-base-v2",
    model_kwargs={"device": "cpu"}
)
```

Store in Chroma:

```python
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./embedding_store"
)
```

---

## 🧠 Retrieval + LLM Pipeline (RAG)

### **Prompt Template**

```python
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict extraction system.\n"
     "RULES:\n"
     "1. If the context contains a 'Total', 'Grand Total', or similar value, use it directly.\n"
     "2. DO NOT calculate, add, infer, or recompute totals.\n"
     "3. If total exists, reply: 'The total is already provided: <value>'.\n"
     "4. If missing, reply: 'Total not provided in the context.'\n\n"
     "Context:\n{context}"
    ),
    ("human", "{query}")
])
```

### **LLM Setup**

```python
llm = ChatGroq(
    groq_api_key=api_key,
    model="llama-3.1-8b-instant",
    temperature=0.2
)
```

### **Chains**

```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

stuff_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt
)

rag_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=stuff_chain
)
```

### **Run Query**

```python
response = rag_chain.invoke({"query": "Give total cattle count of Bhiwani district."})
print(response["answer"])
```

---

## 🧪 Example Queries

- “List all goshala names in Hisar district.”
- “What is the total cattle count for Rohtak?”
- “Give the registration numbers of all gaushalas in Fatehabad.”

---

## 🐞 Known Issues & Fixes

### ❗ Problem: Model recalculates totals  
**Fix:** Added strict rules in system prompt.

### ❗ Long tables cause poor embeddings  
**Fix:** Use E5-base-v2 + recursive splitter.

### ❗ Retrieval brings unrelated districts  
**Fix:** Use:
```python
search_kwargs={"k": 3, "score_threshold": 0.35}
```

---

## 📌 Future Enhancements

- Replace PDF → text with direct table → JSON extraction.
- Improve chunking by merging rows by district.
- Build a frontend using FastAPI + React.

---

## 📧 Contact

For help or guidance, feel free to ask!

