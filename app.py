import os
import gradio as gr
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Load the GROQ API KEY
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

llm = ChatGroq(temperature=0, model_name='llama-3.1-8b-instant', groq_api_key=GROQ_API_KEY)

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
</context>
Question: {input}
"""
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectors = None

def process_pdf(file):
    global vectors
    if file is not None:
        loader = PyPDFLoader(file.name)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)
        if vectors is None:
            vectors = FAISS.from_documents(final_documents, embeddings)
        else:
            vectors.add_documents(final_documents)
        return "PDF processed and added to the knowledge base."
    return "No file uploaded."

def process_question(question):
    if vectors is None:
        return "Please upload a PDF first.", "", 0

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({'input': question})
    
    context = "\n\n".join([doc.page_content for doc in response["context"]])
    
    # Calculate a simple confidence score based on the relevance of retrieved documents
    confidence_score = sum([doc.metadata.get('score', 0) for doc in response["context"]]) / len(response["context"])
    
    return response['answer'], context, round(confidence_score, 2)

CSS = """
.duplicate-button { 
    margin: auto !important; 
    color: white !important; 
    background: black !important; 
    border-radius: 100vh !important;
}
h3, p, h1 { 
    text-align: center; 
    color: white;
}
footer { 
    text-align: center; 
    padding: 10px; 
    width: 100%; 
    background-color: rgba(240, 240, 240, 0.8); 
    z-index: 1000; 
    position: relative; 
    margin-top: 10px; 
    color: black;
}
"""

FOOTER_TEXT = """
<footer>
    <p>If you enjoyed the functionality of the app, please leave a like!<br>
    Check out more on <a href="https://www.linkedin.com/in/your-linkedin/" target="_blank">LinkedIn</a> | 
    <a href="https://your-portfolio-url.com/" target="_blank">Portfolio</a></p>
</footer>
"""

TITLE = "<h1>📚 RAG Document Q&A 📚</h1>"

with gr.Blocks(css=CSS, theme="Nymbo/Nymbo_Theme") as demo:
    gr.HTML(TITLE)

    with gr.Tab("PDF Uploader"):
        pdf_file = gr.File(label="Upload PDF")
        upload_button = gr.Button("Process PDF")
        upload_output = gr.Textbox(label="Upload Status")

    with gr.Tab("Q&A System"):
        question_input = gr.Textbox(lines=2, placeholder="Enter your question here...")
        submit_button = gr.Button("Ask Question")
        answer_output = gr.Textbox(label="Answer")
        context_output = gr.Textbox(label="Relevant Context", lines=10)
        confidence_output = gr.Number(label="Confidence Score")

    upload_button.click(process_pdf, inputs=[pdf_file], outputs=[upload_output])
    submit_button.click(process_question, inputs=[question_input], outputs=[answer_output, context_output, confidence_output])

    gr.HTML(FOOTER_TEXT)

if __name__ == "__main__":
    demo.launch()
