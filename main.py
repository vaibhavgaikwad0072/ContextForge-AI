from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

Embedding_model= HuggingFaceEmbeddings()

vectorestore = Chroma(
    persist_directory="chroma-db",
    embedding_function=Embedding_model
)

retriever = vectorestore.as_retriever(
    search_type = "mmr",
    search_kwargs={
        "k":4,
        "fetch_k":10,
        "lambda_mult":0.5
    }
)

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.5
)

promt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document.
         
"""),("human",
      """Context:
{context}

Question:
{question}
""")
    ]
)

print("--------------Welcome--------------")
print()
print("Exit Press 0")

while True:
    query = input("YOU : ")
    if query=="0":
        break

    docs = retriever.invoke(query)

    context ="\n\n".join([doc.page_content for doc in docs])

    final_promt=promt.invoke(
        {
            "context":context,
            "question":query
        }
    )

    response = model.invoke(final_promt)

    print("\n AI : " , response.content)
