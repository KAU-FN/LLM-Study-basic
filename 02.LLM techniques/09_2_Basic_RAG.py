# Standard library
import logging
import sys
import os

# Third-party import
from IPython.display import Markdown, display

# Qdrant client
import qdrant_client

# LlamaIndedx core
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings

# LlamaIndex vector store
from llama_index.vector_stores.qdrant import QdrantVectorStore

# Embedding model
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

# LLM
from llama_index.llms.google_genai import GoogleGenAI

# API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# Base LLM setting
Settings.llm = GoogleGenAI(
    model = 'gemini-2.5-flash',
    api_key = GEMINI_API_KEY
)

Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Data loading
from llama_index.core import Document

reader = SimpleDirectoryReader(os.getcwd()+'/rag_dataset', recursive=True)
documents = reader.load_data(show_progress=True)

# Vector DB setup
client = qdrant_client.QdrantClient(
    path = './rag_dataset'
)

vector_store = QdrantVectorStore(client=client, collection_name="01_Basic_RAG")

# Vector DB에 데이터 넣기
# Set up ingestion pipeline

# from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.node_parser import SentenceSplitter
# from llama_index.core.node_parser import MarkdownNodeParser
# from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.ingestion import IngestionPipeline

# # 파이프라인 통해서 안에 있는 문서들을 chunk 단위로 분할
# # 이때 embedding 하는 모델은 앞서 선언한 embed_model을 사용
# pipeline = IngestionPipeline(
#     transformations = [
#         SentenceSplitter(chunk_size=1024, chunk_overlap=20),
#         Settings.embed_model
#     ],
#     vector_store = vector_store
# )

# # vector DB에 chunk들을 넣기
# nodes = pipeline.run(documents=documents, show_progress=True)
# print("Number of chunks added to vector DB :",len(nodes))

# 유사도 검색 indexing을 위한 준비
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# Prompt 수정 및 tuning
from llama_index.core import ChatPromptTemplate

qa_prompt_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {query_str}\n"
)

refine_prompt_str = (
    "We have the opportunity to refine the original answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question: {query_str}. "
    "If the context isn't useful, output the original answer again.\n"
    "Original Answer: {existing_answer}"
)

# Text QA Prompt
chat_text_qa_msgs = [
    ("system","You are a AI assistant who is well versed with answering questions from the provided context"),
    ("user", qa_prompt_str),
]
text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)

# Refine Prompt
chat_refine_msgs = [
    ("system","Always answer the question, even if the context isn't helpful.",),
    ("user", refine_prompt_str),
]
refine_template = ChatPromptTemplate.from_messages(chat_refine_msgs)

# Simple chat application
from typing import List
from llama_index.core.base.llms.types import ChatMessage, MessageRole

class ChatEngineInterface:
    def __init__(self, index):
        self.chat_engine = index.as_chat_engine()
        self.chat_history: List[ChatMessage] = []

    def display_message(self, role:str, content:str):
        if role == "USER":
            # display(Markdown(f"**Human:** {content}"))
            print(f"HUMAN: {content}")
        else:
            # display(Markdown(f"**AI:** {content}"))
            print(f"AI: {content}")
    
    def chat(self, message:str) -> str:
        # user input을 위한 ChatMessage 생성
        user_message = ChatMessage(role=MessageRole.USER, content=message)
        self.chat_history.append(user_message)

        # chat engine에서 response 받기
        response = self.chat_engine.chat(message, chat_history=self.chat_history)

        # AI 응답을 위한 ChatMessage 생성
        ai_message = ChatMessage(role=MessageRole.ASSISTANT, content=str(response))
        self.chat_history.append(ai_message)

        # 대화 내역 display
        self.display_message("USER", message)
        self.display_message("ASSITANT", str(response))

        return str(response)
    
    def get_chat_history(self) -> List[ChatMessage]:
        return self.chat_history

chat_interface = ChatEngineInterface(index)
# while True:
#     user_input = input("You: ").strip()

#     if user_input.lower() == 'exit':
#         print("Goodbye!")
#         break
    
#     chat_interface.chat(user_input)

# Gradio 사용
import gradio as gr
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import qdrant_client
import os
import tempfile
import shutil
from typing import List
from llama_index.core.base.llms.types import ChatMessage, MessageRole

class RAGChatbot:
    def __init__(self):
        self.client = qdrant_client.QdrantClient(path="./Demo_RAG")
        self.vector_store = None
        self.index = None
        self.chat_engine = None
        self.chat_history = []
        # Initialize vector store and index
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name="Demo_RAG"
        )

        # Create the index and ingest documents
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store
        )

        # Initialize chat engine
        self.chat_engine = self.index.as_chat_engine(
            streaming=True,
            verbose=True
        )


    def process_uploaded_files(self, files) -> str:
        try:
            # Create a temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded files to temporary directory
                for file in files:
                    shutil.copy(file.name, temp_dir)

                # Load documents
                reader = SimpleDirectoryReader(temp_dir)
                documents = reader.load_data()

                pipeline = IngestionPipeline(
                    transformations=[
                        # MarkdownNodeParser(include_metadata=True),
                        # TokenTextSplitter(chunk_size=500, chunk_overlap=20),
                        SentenceSplitter(chunk_size=1024, chunk_overlap=20),
                        # SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95 , embed_model=Settings.embed_model),
                        Settings.embed_model,
                    ],
                    vector_store=self.vector_store,
                )

                # Ingest directly into a vector db
                nodes = pipeline.run(documents=documents , show_progress=True)

                return f"Successfully processed {len(documents)} documents. Ready to chat! and inserted {len(nodes)} into the database"

        except Exception as e:
            return f"Error processing files: {str(e)}"

    def chat(self, message: str, history: List[List[str]]) -> List[List[str]]:
        if self.chat_engine is None:
            return history + [[message, "Please upload documents first before starting the chat."]]

        try:
            # Convert history to ChatMessage format
            chat_history = []
            for h in history:
                chat_history.extend([
                    ChatMessage(role=MessageRole.USER, content=h[0]),
                    ChatMessage(role=MessageRole.ASSISTANT, content=h[1])
                ])

            # Add current message to history
            chat_history.append(ChatMessage(role=MessageRole.USER, content=message))

            # Get response from chat engine
            response = self.chat_engine.chat(message, chat_history=chat_history)

            # Return the updated history with the new message pair
            return history + [[message, str(response)]]

        except Exception as e:
            return history + [[message, f"Error generating response: {str(e)}"]]

def create_demo():
    # Initialize the chatbot
    chatbot = RAGChatbot()

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# RAG Chatbot")
        gr.Markdown("Upload your documents and start chatting!")

        with gr.Row():
            with gr.Column(scale=1):
                file_output = gr.File(
                    file_count="multiple",
                    label="Upload Documents",
                    file_types=[".txt", ".pdf", ".docx", ".md"]
                )
                upload_button = gr.Button("Process Documents")
                status_box = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=2):
                chatbot_interface = gr.Chatbot(
                    label="Chat History",
                    height=400,
                    bubble_full_width=False,
                )
                with gr.Row():
                    msg = gr.Textbox(
                        label="Type your message",
                        placeholder="Ask me anything about the uploaded documents...",
                        lines=2,
                        scale=4
                    )
                    submit_button = gr.Button("Submit", scale=1)
                clear = gr.Button("Clear")

        # Event handlers
        upload_button.click(
            fn=chatbot.process_uploaded_files,
            inputs=[file_output],
            outputs=[status_box],
        )
        submit_button.click(
            fn=chatbot.chat,
            inputs=[msg, chatbot_interface],
            outputs=[chatbot_interface],
        )

        clear.click(
            lambda: None,
            None,
            chatbot_interface,
            queue=False
        )

    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True)
    