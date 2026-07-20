import io
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline


app = FastAPI(title="AI Image Generator Backend")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

print("Loading Stable Diffusion model...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    torch_dtype=torch_dtype
).to(device)
print("Model loaded successfully!")


class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = "blurry, low quality, distorted"
    steps: int = 30
    guidance: float = 7.5

@app.post("/generate")
async def generate_image(request: ImageRequest):
    try:
        
        result = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance
        )
        
        image = result.images[0]
        
        
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        
        return StreamingResponse(img_byte_arr, media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="127.0.0.1", port=8000)