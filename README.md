# Image Caption Generator (CNN + LSTM)

A deep learning multimodal system combining computer vision feature extraction with recurrent natural language sequence modeling to automatically generate rich descriptive captions for any input image.

## Architecture
- **Vision Backbone**: Deep convolutional feature encoder extracting spatial semantic feature maps.
- **Language Decoder**: Long Short-Term Memory (LSTM) network with attention mechanism and beam search decoding (width k=3).
- **Validation**: Evaluated on MS-COCO / Flickr8k benchmark with BLEU-4 metrics.

## Quickstart
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
python app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) and drag-and-drop an image.
