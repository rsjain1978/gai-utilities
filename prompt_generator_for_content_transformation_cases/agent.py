import os
import asyncio
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from openai import AsyncOpenAI
from termcolor import colored
from dotenv import load_dotenv
import webbrowser
import uvicorn

# Load environment variables
load_dotenv()

app = FastAPI()

# Create static folder if it doesn't exist
if not os.path.exists("static"):
    os.makedirs("static")

# Set up static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize OpenAI client
client = AsyncOpenAI()

async def llm_call(messages, model="gpt-4o-mini"):
    """Generic async function for LLM calls"""
    try:
        print(colored(f"Making LLM call with model: {model}", "cyan"))
        response = await client.chat.completions.create(
            model=model,
            messages=messages
        )
        print(colored("LLM call successful", "green"))
        return response.choices[0].message.content
    except Exception as e:
        print(colored(f"Error in LLM call: {str(e)}", "red"))
        raise

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Render the main page"""
    try:
        return templates.TemplateResponse(
            "index.html",
            {"request": request}
        )
    except Exception as e:
        print(colored(f"Error rendering index page: {str(e)}", "red"))
        raise

def open_browser():
    """Open the browser when the server starts"""
    webbrowser.open("http://localhost:8000")

if __name__ == "__main__":
    # Open browser after a short delay
    asyncio.get_event_loop().run_in_executor(None, lambda: asyncio.run(asyncio.sleep(1.5)) or open_browser())
    
    # Run the server
    uvicorn.run("agent:app", host="0.0.0.0", port=8000, reload=True) 