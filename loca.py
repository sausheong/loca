import os
from dotenv import load_dotenv, find_dotenv

from threading import Thread
from typing import Any
from queue import Queue

from flask import Flask, render_template, request, Response
from waitress import serve

from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms import LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager

load_dotenv(find_dotenv())

# use a threaded generator to return response in a stream
class ThreadedGenerator:
    def __init__(self):
        self.queue = Queue()

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is StopIteration: raise item
        return item

    def send(self, data):
        self.queue.put(data)

    def close(self):
        self.queue.put(StopIteration)

# a callback handler to send the tokens to the generator
class LocaStreamingCallbackHandler(StreamingStdOutCallbackHandler):
    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.gen.send(token)


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

# setup the memory
memory = ConversationBufferWindowMemory(memory_key="history", k=3)

# setup the thread to run the LLM query
def llm_thread(g, query):
    loca_callback = LocaStreamingCallbackHandler(g)
    try:
        llm = LlamaCpp(
            model_path=os.getenv('LOCAL_MODEL'),
            n_gpu_layers=38,
            n_batch=2048,
            max_tokens=2048,
            f16_kv=True,
            temperature=0,
            n_ctx=4096,                
            streaming=True,
            callback_manager=CallbackManager([loca_callback]),
        )        
        llm_chain = LLMChain(llm=llm, 
                            prompt=prompt, 
                            memory=memory,
                            verbose=True)        
        llm_chain.run(query)
    finally:
        g.close()

# run the query in a thread
def llm_run(query):
    g = ThreadedGenerator()
    Thread(target=llm_thread, args=(g, query)).start()
    return g

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
    return Response(llm_run(data['input']), mimetype='text/event-stream')    
    

if __name__ == '__main__':
    print("\033[93mLoca started. Press CTRL+C to quit.\033[0m")
    serve(loca, port=1339, threads=16)
