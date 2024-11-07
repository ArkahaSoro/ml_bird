#установка библиотек
import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

#загрузка моделей
@st.cache(allow_output_mutation = True)
def load_auto_image_processor():
  return AutoImageProcessor.from_pretrained('chrimaue/bird-species-classifier')

@st.cache(allow_output_mutation = True)
def load_auto_model():
  return AutoModelForImageClassification.from_pretrained('chrimae/bird-species-classifer')

#инициализация процессора и модели
processor = load_auto_image_processor()
model = load_auto_model()

#реализация функции распознания объектов на изображении
def predict_step(image):
  if image.mode != "RGB":
    image = image.convert(mode="RGB")
pixel_values = processor(images = image, return_tensors="pt").pixel_values
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pixel_values = pixel_values.to(device)
model.to(device)
with torch.no_grad():
  outputs = model(pixel_values)
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
labels = processor.config.id2label
predicted_label = labels[predicted_class_idx]
return predicted_label

#использование в стримлит
st.title("Bird Species Classifier")
uploaded_file = st.file_uploader("Выбирает изображение...", type =["jpg","jpeg","png"])

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  st.image(image, caption='Upload Image.', use_column_width = True)
  st.write("")
  st.write("определяет вид...")

  prediction = predict_step(image)
  st.write(f"Относится к виду:{prediction}")
