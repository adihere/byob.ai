import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

app = Flask(__name__)

# load env variables
load_dotenv()
openai_org = os.getenv('OPENAI_API_ORG')
openai_api_key1 = os.getenv('OPENAI_API_KEY1')

# load docs
directory = './content'


def load_docs(directory):
  loader = DirectoryLoader(directory, loader_cls=TextLoader)
  documents = loader.load()
  return documents


documents = load_docs(directory)


#split docs
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                 chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs


docs = split_docs(documents)

#embed
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

#create vector db
from langchain.vectorstores import Chroma
#db = Chroma.from_documents(docs, embeddings)

# Specify the directory where the Chroma database should be persisted.
persist_directory = "./db"

try:
  db = Chroma.from_documents(docs,
                             embeddings,
                             persist_directory=persist_directory)
  # Persist the database to disk.
  db.persist()
except Exception as e:
  print(f"An error occurred while persisting the Chroma database: {e}")

# Using OpenAI Large Language Models (LLM) with Chroma DB
os.environ["OPENAI_API_KEY"] = openai_api_key1

#assign LLM model for use
from langchain.chat_models import ChatOpenAI

model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name)

# Extracting Answers from Documents
from langchain.chains.question_answering import load_qa_chain

chain = load_qa_chain(llm, chain_type="stuff", verbose=True)


@app.route('/chatbot', methods=['POST'])
def chatbot():
  # Get the user message from the request
  user_message = request.json['message']

  # Generate a response using the chatbot code
  response = generate_response(user_message)

  # Return the response as JSON
  return jsonify(response)


def generate_response(user_message):
  # Use the existing code to generate a response to the user message
  matching_docs = db.similarity_search(user_message)
  answer = chain.run(input_documents=matching_docs, question=user_message)

  #return answer
  return answer


if __name__ == '__main__':
  # Print the endpoint URL in the terminal
  print(f'Chatbot endpoint: http://{host}:{port}/chatbot')

  # Start the Flask app
  app.run(host=os.getenv(‘REPLIT_HOST’), port=os.getenv(‘REPLIT_PORT’))
  
