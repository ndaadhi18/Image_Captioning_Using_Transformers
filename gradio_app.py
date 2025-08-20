import gradio as gr
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

model = VisionEncoderDecoderModel.from_pretrained("fine-tuned-captioning-epoch10")
feature_extractor = ViTImageProcessor.from_pretrained("fine-tuned-captioning-epoch10")
tokenizer = AutoTokenizer.from_pretrained("fine-tuned-captioning-epoch10")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def generate_caption(image):
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)

    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption


interface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🖼️ Image Captioning using Transformer",
    description="Upload an image and get a caption from your fine-tuned model."
)

interface.launch()
