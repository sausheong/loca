import os
import webbrowser
from flask import Flask, render_template, request, jsonify
from waitress import serve
import os
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv, find_dotenv
from langchain.llms import CTransformers
from langchain import PromptTemplate, LLMChain

# setup the prompt
template = """You are an AI assistant answering questions from a human. Only
answer the question and not add any other conversation. Do not ask questions
at the end of the answer.
Current conversation:
{history}

Human: {input}
AI: 
"""
prompt = PromptTemplate(template=template, input_variables=["history","input"])

# setup the LLM
load_dotenv(find_dotenv())
llm = CTransformers(
    model=os.getenv('LOCAL_MODEL'), 
    model_type=os.getenv('LOCAL_MODEL_TYPE'), 
    config={'context_length': 1024, 'max_new_tokens': 2048, 'temperature': 0.7},
)

# setup the memory
memory = ConversationBufferWindowMemory(memory_key="history", k=3)

# create an LLM chain
chain = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)

# get path for static files
static_dir = os.path.join(os.path.dirname(__file__), 'static')
if not os.path.exists(static_dir):
    static_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'static')

# start server
print("\033[96mStarting Loca at http://127.0.0.1:1339\033[0m")
loca = Flask(__name__, static_folder=static_dir, template_folder=static_dir)

# server landing page
@loca.route('/')
def landing():
    return render_template('index.html')

# run
@loca.route('/run', methods=['POST'])
def run():
    data = request.json
    response = chain.run(data['input'])
    return jsonify({'input': data['input'],
                    'response': response})

if __name__ == '__main__':
    print("\033[93mLoca started. Press CTRL+C to quit.\033[0m")
    webbrowser.open("http://127.0.0.1:1339")
    serve(loca, port=1339, threads=16)
