import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

app = FastAPI()

@app.get('/')
async def main():
    return "Hello World"

if __name__ == '__main__':
    print("Running on", os.getenv("UVICORN_PORT"))
    uvicorn.run(
        app="app.api:app",
        host=os.getenv("UVICORN_HOST"),
        port=int(os.getenv("UVICORN_PORT"))
    )

