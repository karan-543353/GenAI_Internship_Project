import torch
from diffusers import StableDiffusionPipeline
# 1. Check for GPU availability
# If you have an NVIDIA GPU, this will use it to speed up generation significantly.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# If using a GPU, load in 16-bit float to save memory. For CPU, use default float32.
torch_dtype = torch.float16 if device == "cuda" else torch.float32

# 2. Load the Pre-trained Diffusion Model
print("Loading the model... (This may take a few minutes the first time to download weights)")
model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch_dtype
)

# Move the pipeline to your selected device
pipe = pipe.to(device)

# 3. Define your prompt
prompt = input(" Type the Prompt here -------- ")
negative_prompt = "blurry, low quality, distorted, ugly"

# 4. Generate the image
print(f"Generating image for prompt: '{prompt}'...")

# The generator ensures we get different results each time, or the same if we set a manual seed
generator = torch.Generator(device).manual_seed(42) 

# Run the diffusion process
result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30, # Higher steps = better quality but slower (30-50 is standard)
    guidance_scale=7.5,     # How closely the model should follow the prompt (7-10 is good)
    generator=generator
)

# Extract the generated image
image = result.images[0]

# 5. Save and display the image
output_filename = "generated_image.png"
image.save(output_filename)
print(f"Success! Image saved as {output_filename}")

# Optionally display it if your environment supports it
image.show()