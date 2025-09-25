import streamlit as st 
from rag_pipeline import answer_query, retireve_docs, llm_model

st.title("Upload your file")
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    st.success(f"File '{uploaded_file.name}' uploaded successfully!")






#STEP: 2 : Chatbot skeleton (question & Answer)


user_query = st.text_area("Enter your promt: " , height=150,placeholder="Ask your question here...")

ask_question = st.button("Ask From Vaibhav's chatbot")

if(ask_question):

    if uploaded_file:
        st.chat_message("user").write(user_query)

        #RAG PIPELINE
        retireve_docs=retireve_docs(user_query)
        response=answer_query(documents=retireve_docs, model=llm_model, query=user_query)
        #fixed_response = "Hey, this is a sample response from Vaibhav's chatbot."
        st.chat_message("VAIBHAV").write(response)

    else:
        st.error("Please upload a file to ask questions.")