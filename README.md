# RAG Document Q&A

To know more, check out my blog - [Enhancing Q&A with RAG Technology](https://medium.com/@girishwangikar/enhancing-q-a-with-rag-technology-501454c009ad)

RAG Document Q&A is an interactive application designed to answer questions based on the content of uploaded PDF documents. Leveraging the llama-3.1-8B-Instant model and FAISS, this app provides accurate and context-aware responses, making it ideal for document-based Q&A tasks.

## Features

- **Advanced Language Model**: Utilizes the llama-3.1-8B-Instant model for generating intelligent responses based on the provided context.
- **Document-Based Q&A**: Upload PDF documents and ask questions directly related to the content.
- **Real-Time Response Streaming**: Get instant answers with context and confidence scores.
- **User-Friendly Interface**: Built with Gradio for an intuitive and easy-to-use web interface.
- **PDF Processing**: Automatically processes and splits PDF documents into manageable chunks for better retrieval.
- **Stylish UI**: Comes with a dark mode theme for comfortable usage.

## Prerequisites

Before running this application, ensure you have the following:

- Python 3.7+
- Gradio
- LangChain
- FAISS (installed with LangChain)
- HuggingFace Transformers

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/GirishWangikar/RAG-Document-Q-A-System
    cd rag-document-qa
    ```

2. Install the required packages:
    ```bash
    pip install gradio langchain-groq 
    ```

3. Set up your Groq API key as an environment variable:
    ```bash
    export API_KEY='your_api_key_here'
    ```

## Usage

1. Run the application:
    ```bash
    python app.py
    ```

2. Open your web browser and navigate to the URL provided in the console output.

3. Start by uploading a PDF document using the "PDF Uploader" tab.

4. After processing the PDF, navigate to the "Q&A System" tab, type your question, and click "Ask Question" to get the answer, relevant context, and a confidence score.

## Customization

- **System Prompt**: Modify the default system prompt to change the AI's persona or behavior.
- **Temperature**: Adjust the randomness of the AI's responses (0 for more deterministic, 1 for more creative).
- **Max Tokens**: Set the maximum length of the AI's responses.

## Contact

Created by Girish Wangikar

Check out more on [LinkedIn](https://www.linkedin.com/in/girish-wangikar/) | [Portfolio](https://girishwangikar.github.io/Girish_Wangikar_Portfolio.github.io/) | [Technical Blog - Medium](https://medium.com/@girishwangikar/enhancing-q-a-with-rag-technology-501454c009ad)

