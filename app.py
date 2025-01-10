from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


app = Flask(__name__)
CORS(app) 

# file_path = "sample_dataset.csv"

# loader = CSVLoader(file_path=file_path)
# data = loader.load()

# for record in data[:2]:
#     print(record)

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
# docs = text_splitter.split_documents(data)
# print(docs)

embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
# print(embed_model)

# vectorstore = Chroma.from_documents(documents=docs, embedding=embed_model, persist_directory="./chroma_db")
# print(vectorstore)

vectorstore_disk = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embed_model
)

retriever = vectorstore_disk.as_retriever(search_type="similarity", search_kwargs={"k": 10})
print(retriever)

groq_api_key="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"
llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.1-8b-instant")



system_prompt = (
    "You are an assistant for question-answering tasks."
    "Use the following pieces of retrieved context to answer"
    "If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        # Get user query from request
        data = request.json
        user_query = data.get('question', '')

        if not user_query:
            return jsonify({"error": "Question is required."}), 400

        # Create the question-answer chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Generate response
        response = rag_chain.invoke({"input": user_query})
        answer = response.get("answer", "I don't know the answer to that.")

        return jsonify({"answer": answer}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True, port=5000)