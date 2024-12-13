import os
import uvicorn
import gradio as gr
from dotenv import load_dotenv
from utils.inference import evaluate
from utils.criteria import get_criteria
from api import FastAPI
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

class Request(BaseModel):
    writing_sample : str
    writing_question : str

class Response(BaseModel):
    evaluation : dict

@app.post("/",response_model=Response)
async def evaluate_api(prompt:Request):
    criteria_string = get_criteria()
    try:
        response = evaluate(prompt.writing_sample, prompt.writing_question, criteria_string)
        return {"evaluation": response}
    except Exception as e:
        return {"error": str(e)}

demo = gr.Interface(
    fn=evaluate,
    inputs=[
        gr.Textbox(label="Question", lines=6),
        gr.Textbox(label="Answer", lines=6),
    ],
    outputs=gr.Textbox(label="Output"),
    title="Writing Test Evaluation App",
    description="Enter the question you attempted and the answer in their respective boxes."
)

app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == '__main__':
    uvicorn.run(
        app="main:app",
        host=os.getenv("UVICORN_HOST"),
        port=int(os.getenv("UVICORN_PORT"))
    )

