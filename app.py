"""
Multilingual Text-to-Image Generator
Type a prompt in any language -> translated to English -> Stable Diffusion 2 -> image.

Run:  python app.py   then open http://127.0.0.1:5000
"""
import base64
import io
import os
import threading

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from flask import Flask, jsonify, render_template, request
from langdetect import DetectorFactory, detect
from transformers import MarianMTModel, MarianTokenizer

DetectorFactory.seed = 0  # make language detection deterministic

# - Configuration -
#MODEL_ID = os.environ.get("MODEL_ID", "stabilityai/stable-diffusion-2-1-base")
MODEL_ID = os.environ.get("MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5")
TRANSLATOR_ID = "Helsinki-NLP/opus-mt-mul-en"  # many languages -> English

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# - Load models once at startup -
print(f"Device: {DEVICE}")
print("Loading translation model...")
tr_tokenizer = MarianTokenizer.from_pretrained(TRANSLATOR_ID)
tr_model = MarianMTModel.from_pretrained(TRANSLATOR_ID)

print("Loading Stable Diffusion (first run downloads several GB)...")
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(DEVICE)
pipe.enable_attention_slicing()  # lowers memory use
print("Ready.")

app = Flask(__name__)
gpu_lock = threading.Lock()  # one generation at a time


def to_english(text: str) -> tuple[str, str]:
    """Return (english_text, detected_language_code)."""
    try:
        lang = detect(text)
    except Exception:
        lang = "unknown"
    if lang == "en":
        return text, lang
    batch = tr_tokenizer([text], return_tensors="pt", padding=True)
    out = tr_model.generate(**batch, max_new_tokens=128)
    return tr_tokenizer.decode(out[0], skip_special_tokens=True), lang


@app.route("/")
def index():
    return render_template("index.html", device=DEVICE)


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify(error="Enter a description first."), 400
    if len(prompt) > 500:
        return jsonify(error="Keep the description under 500 characters."), 400

    # On CPU, cap quality settings so it finishes in a reasonable time.
    default_steps = 25 if DEVICE == "cuda" else 12
    steps = max(5, min(int(data.get("steps", default_steps)), 50))
    size = 512 # if DEVICE == "cuda" else 384
    seed = data.get("seed")

    try:
        english, lang = to_english(prompt)
        generator = None
        if seed not in (None, ""):
            generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

        with gpu_lock:
            image = pipe(
                english,
                num_inference_steps=steps,
                guidance_scale=7.5,
                height=size,
                width=size,
                generator=generator,
            ).images[0]

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode()
        return jsonify(
            image=f"data:image/png;base64,{encoded}",
            english_prompt=english,
            language=lang,
        )
    except Exception as exc:  # show a readable error in the UI
        return jsonify(error=f"Generation failed: {exc}"), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)


