from flask import Flask, render_template, request, jsonify, send_file
from diffusers import StableDiffusionPipeline
import torch
import os
from datetime import datetime
from PIL import Image
import io
import threading
import csv

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "generated_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.config["UPLOAD_FOLDER"] = OUTPUT_DIR

LOG_FILE = os.path.join(OUTPUT_DIR, "generation_log.csv")

# Initialize CSV log file if it doesn't exist
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Timestamp', 'Prompt', 'Enhanced_Prompt', 'Negative_Prompt', 
            'Steps', 'Width', 'Height', 'Num_Images', 'Filenames', 'Device'
        ])

# Global state
pipe = None
model_ready = False

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "dreamlike-art/dreamlike-diffusion-1.0"

def load_model():
    global pipe, model_ready
    print("=" * 60)
    print("IMAGIRI: Loading Stable Diffusion model...")
    print(f"📱 Device: {device.upper()}")
    print("=" * 60)
    
    try:
        pipe_local = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=True,
        ).to(device)
        pipe = pipe_local
        model_ready = True
        print("IMAGIRI: Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model_ready = False

# Load model in background thread
loader_thread = threading.Thread(target=load_model, daemon=True)
loader_thread.start()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def status():
    return jsonify({"ready": model_ready, "device": device})

def log_generation(prompt, enhanced_prompt, negative_prompt, steps, width, height, num_images, filenames):
    """Log image generation to CSV file"""
    try:
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                prompt,
                enhanced_prompt if enhanced_prompt else "",
                negative_prompt,
                steps,
                width,
                height,
                num_images,
                ";".join(filenames),
                device
            ])
    except Exception as e:
        print(f"❌ Error logging to CSV: {e}")

@app.route("/api/generate", methods=["POST"])
def generate_image():
    global pipe, model_ready
    
    if not model_ready or pipe is None:
        return jsonify({"success": False, "error": "Model is still loading. Please wait..."}), 503

    data = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()
    negative_prompt = data.get("negative_prompt", "").strip()
    enhancement = bool(data.get("enhancement", False))

    if not prompt:
        return jsonify({"success": False, "error": "Prompt cannot be empty."}), 400

    original_prompt = prompt
    if enhancement:
        prompt += ", highly detailed, 4K quality, professional photography, sharp focus, masterpiece"

    params = {
        "num_inference_steps": int(data.get("steps", 10)),
        "width": int(data.get("width", 512)),
        "height": int(data.get("height", 640)),
        "num_images_per_prompt": int(data.get("num_images", 1)),
        "negative_prompt": negative_prompt if negative_prompt else None,
    }
    params = {k: v for k, v in params.items() if v is not None}

    try:
        print(f"\nGenerating image...")
        print(f"   Prompt: {prompt}")
        print(f"   Steps: {params['num_inference_steps']}, Size: {params['width']}x{params['height']}")
        
        with torch.no_grad():
            result = pipe(prompt, **params)
            images = result.images

        saved_images = []
        filenames_list = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for idx, img in enumerate(images):
            base_name = f"imagiri_{timestamp}_{idx}"
            filename = f"{base_name}.png"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            img.save(filepath)
            filenames_list.append(filename)

            saved_images.append(
                {
                    "filename": filename,
                    "url_png": f"/download/{base_name}/png",
                    "url_jpg": f"/download/{base_name}/jpg",
                }
            )

        # Log to CSV
        log_generation(
            original_prompt, 
            prompt if enhancement else None, 
            negative_prompt, 
            params['num_inference_steps'],
            params['width'],
            params['height'],
            params['num_images_per_prompt'],
            filenames_list
        )

        print(f"Generated {len(images)} image(s)")
        print(f"Logged to CSV: generation_log.csv")
        
        return jsonify(
            {
                "success": True,
                "images": saved_images,
                "prompt": original_prompt,
                "enhanced_prompt": prompt if enhancement else None,
                "parameters": params,
                "timestamp": timestamp,
            }
        )

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/download/<basename>/<fmt>")
def download_image(basename, fmt):
    png_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{basename}.png")
    
    if not os.path.exists(png_path):
        return "File not found", 404

    img = Image.open(png_path).convert("RGB")
    buf = io.BytesIO()

    if fmt.lower() in ["jpg", "jpeg"]:
        img.save(buf, format="JPEG", quality=95)
        mimetype = "image/jpeg"
        ext = "jpg"
    else:
        img.save(buf, format="PNG")
        mimetype = "image/png"
        ext = "png"

    buf.seek(0)
    download_name = f"{basename}.{ext}"
    
    return send_file(
        buf,
        mimetype=mimetype,
        as_attachment=True,
        download_name=download_name,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
