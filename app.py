from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from llama_cpp import Llama
import os

# Create FastAPI app
app = FastAPI(title="Description Generator")

# Model path - will look for the model file in the models directory
MODEL_PATH = os.environ.get("MODEL_PATH", "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

# Load the model (this will happen when the app starts)
model = Llama(
    model_path=MODEL_PATH,
    n_ctx=512,  # Smaller context window to save memory
    n_batch=8,  # Smaller batch size for memory efficiency
)

# Define the request structure
class DescriptionRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

# Create the generate endpoint
@app.post("/generate")
async def generate_description(request: DescriptionRequest):
    # Format the prompt
    formatted_prompt = f"Generate a description for: {request.prompt}\nDescription:"
    
    # Generate the description
    output = model(
        formatted_prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    
    # Return the generated text
    return {"description": output["choices"][0]["text"]}

# Create a simple homepage
@app.get("/")
async def root():
    return {"message": "Description Generator is running! Send POST requests to /generate endpoint."}

# Run the app if this file is executed directly
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)