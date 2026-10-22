# Multilingual Text-to-Image Generator

Prompt in any language -> translated to English (Helsinki-NLP opus-mt-mul-en)
-> Stable Diffusion 2.1 -> image shown in the browser.

## Setup
    python -m venv venv
    venv\Scripts\activate (Windows)
    pip install -r requirements.txt
    python app.py

Open http://127.0.0.1:5000

## Notes
- First run downloads several GB of models. Be patient.
- With an NVIDIA GPU: 512px, 25 steps, a few seconds per image.
- Without a GPU: 384px, 12 steps, roughly 1-5 minutes per image.
  For speed, run it on Google Colab with a free GPU.
- If the model download is refused, set a different model:
    set MODEL_ID=your/model-id      (Windows)
- Install a CUDA build of PyTorch from pytorch.org if you have an NVIDIA GPU.
