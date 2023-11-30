import gradio as gr
from langchain.llms import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import os


os.environ['CURL_CA_BUNDLE'] = ''

load_dotenv()


class Document:
    def __init__(self, path):
        self.path = path
        self.text_loader = TextLoader(self.path)
        self.text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=0)
        self.documents = self.text_splitter.split_documents(self.text_loader.load())
        self.sentence_embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = Chroma.from_documents(self.documents, self.sentence_embeddings)

    def get_retriever(self):
        return self.vector_store.as_retriever()


class Chatbot:
    MAIN_PROMPT = """As a warehouse assistant you should help people based on>:{context}. {question}"""

    def __init__(self):
        self.llm = OpenAI(model="gpt-3.5-turbo-instruct", openai_api_key="OPENAI_API_KEY", max_tokens=512)


    def get_chat_prompt_template(self):
        return ChatPromptTemplate.from_template(Chatbot.MAIN_PROMPT)

    def get_model(self):
        return self.llm

    def generate_chain(self, document_instance):
        self.chain = ({"context": document_instance.get_retriever(), "question": RunnablePassthrough()} |
                      self.get_chat_prompt_template() | self.get_model())

    def chatbot_response(self, message, history):
        response = ""
        for chunck in self.chain.stream(message):
            response += chunck
            yield response


class ChatInterface:
    def __init__(self, chatbot):
        self.chatbot = chatbot

    def launch(self):
        chat_interface = gr.ChatInterface(
            self.chatbot.chatbot_response,
            examples=[
                "Who is allowed to operate a lathe? What protective gear should be used to do it?"
            ],
            title="Chatbot",
        )
        chat_interface.launch()


if __name__ == "__main__":
    document = Document("safety.txt")
    chatbot_instance = Chatbot()
    chatbot_instance.generate_chain(document)
    chat_interface_instance = ChatInterface(chatbot_instance)
    chat_interface_instance.launch()



