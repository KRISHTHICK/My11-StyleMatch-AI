import os
import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load reference images
def load_reference_images(folder):
    images = []
    names = []
    for file in os.listdir(folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(folder, file)
            images.append(Image.open(path).convert("RGB"))
            names.append(file)
    return images, names

# Get embeddings
def get_image_embeddings(images):
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    return embeddings.cpu().numpy()

# Streamlit UI
st.set_page_config(page_title="StyleMatch AI", layout="wide")
st.title("👕 StyleMatch AI - Find Similar Fashion Styles")

uploaded_image = st.file_uploader("📸 Upload a fashion photo", type=["jpg", "jpeg", "png"])
if uploaded_image:
    query_image = Image.open(uploaded_image).convert("RGB")
    st.image(query_image, caption="Uploaded Look", use_column_width=True)

    with st.spinner("🔍 Finding similar styles..."):
        ref_images, ref_names = load_reference_images("sample_data")
        all_images = [query_image] + ref_images
        embeddings = get_image_embeddings(all_images)

        query_vec = embeddings[0].reshape(1, -1)
        ref_vecs = embeddings[1:]
        sims = cosine_similarity(query_vec, ref_vecs).flatten()
        top_idx = np.argsort(sims)[::-1][:3]

        st.subheader("🎯 Top Matching Styles")
        for idx in top_idx:
            st.image(ref_images[idx], caption=f"{ref_names[idx]} ({sims[idx]*100:.2f}% match)", width=200)
