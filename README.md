# GenAI Project

In this project we tried to finetune the pretained Vision Transformer Architecture based image captioning model(ViT-GPT2) on a smaller dataset, this github repository includes the training scripts, models, and a Gradio app.

### Key Components:

- `image_captioning.py`: The main script for the image captioning model.
- `gradio_app.py`: A Gradio interface for the image captioning model.
- `dataset/`: Contains the images and captions for training and evaluation.
- `base-captioning-model/`: The base image captioning model.
- `fine-tuned-captioning-epoch*/`: Fine-tuned models from each epoch of training.
