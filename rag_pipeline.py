from langchain_groq import ChatGroq
from vector_database import faiss_db
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

# step 1 : setup LLM (Use Deepseek - R1 model with Groq)
# load_dotenv

llm_model = ChatGroq(model= "deepseek-r1-distill-llama-70b")

#step2: Retrieve Docs

def faiss_db_similarity_search(query):
    return faiss_db.similarity_search(query)

def retireve_docs(query):
    return faiss_db_similarity_search(query)

def get_context(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context

#step 3: ANSWER QUESTION

custom_prompt_template = """
use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
dont provide any additional information other than the answer.
Question: {question}
Context: {context}
Anaswer:
"""

def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    response = chain.invoke({"question": query, "context": context})
    return response

#question = "if a government forbids the right to assemble peacefully which articles are violated and why?"
#retireve_docs=retireve_docs(question)
#print("Vaibhav's Chatbox : ",answer_query(documents=retireve_docs, model=llm_model, query=question))