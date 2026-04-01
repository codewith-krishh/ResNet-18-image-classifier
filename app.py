import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import json

# Page config
st.set_page_config(page_title="Image Classifier", page_icon="🔍", layout="centered")

# Load model (cached so it only loads once per session)
@st.cache_resource
def load_model():
    with open("class_names.json") as f:
        class_names = json.load(f)

    num_classes = len(class_names)

    # Reconstruct EXACT same architecture as training
    model = models.resnet18(weights=None)          # no pretrained weights — we load our own
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # same head replacement

    # Load your fine-tuned weights
    state_dict = torch.load("model.pth", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    return model, class_names

model, class_names = load_model()

# Preprocessing (MUST match training exactly)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Inference function
def predict(image: Image.Image):
    # Convert to RGB — handles PNG with alpha channel (RGBA) or grayscale
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)   # add batch dimension: [1, 3, 224, 224]

    with torch.no_grad():
        outputs = model(tensor)              # raw logits
        probs = F.softmax(outputs, dim=1)    # convert to probabilities

    # Top-2 predictions
    top2_probs, top2_indices = torch.topk(probs, k=min(2, len(class_names)), dim=1)
    results = []
    for prob, idx in zip(top2_probs[0], top2_indices[0]):
        results.append({
            "class": class_names[idx.item()],
            "confidence": prob.item() * 100
        })
    return results

# UI
st.title("🔍 Image Classifier")
st.markdown("Upload an image to get an instant prediction from a fine-tuned ResNet-18.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner("Running inference..."):
            results = predict(image)

        st.markdown("### Prediction")

        # Primary prediction
        top = results[0]
        st.success(f"**{top['class'].upper()}**")
        st.metric(label="Confidence", value=f"{top['confidence']:.1f}%")

        # Top-2 if available
        if len(results) > 1:
            st.markdown("#### Top-2 Breakdown")
            for r in results:
                st.progress(
                    int(r["confidence"]),
                    text=f"{r['class']} — {r['confidence']:.1f}%"
                )

st.markdown("---")
st.caption("Built with PyTorch + ResNet-18 | Fine-tuned on custom dataset")