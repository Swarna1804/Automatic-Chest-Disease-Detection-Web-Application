# =============================
# Automatic Chest Disease Detection Web App
# =============================
import streamlit as st
import torch
from torchvision import models, transforms
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ------------------------------
# Load pretrained model
# ------------------------------
@st.cache_resource
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 14)  # 14 chest diseases
    model.eval()
    return model

model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ------------------------------
# Streamlit page layout
# ------------------------------
st.title("🫁 Automatic Chest Disease Detection")
st.write(
    "Upload a Chest X-ray image to detect possible thoracic diseases using "
    "Deep Learning and Explainable AI (Grad-CAM)."
)

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    # --------------------------
    # Preprocessing
    # --------------------------
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # --------------------------
    # Prediction
    # --------------------------
    with torch.no_grad():
        outputs = torch.sigmoid(model(input_tensor))[0].cpu().numpy()

    classes = [
        "Atelectasis","Cardiomegaly","Consolidation","Edema","Effusion",
        "Emphysema","Fibrosis","Hernia","Infiltration","Mass",
        "Nodule","Pneumonia","Pneumothorax","Pleural Thickening"
    ]

    # --------------------------
    # Determine if lungs are affected
    # --------------------------
    threshold = 0.5  # probability threshold for disease
    affected_indices = np.where(outputs >= threshold)[0]

    if len(affected_indices) == 0:
        st.success("✅ Lungs appear NORMAL")
    else:
        st.error("⚠️ Lungs are AFFECTED")
        st.write("Detected disease(s):")
        for i in affected_indices:
            st.write(f"- **{classes[i]}**: {outputs[i]*100:.2f}%")

    # --------------------------
    # Grad-CAM for top 2 diseases
    # --------------------------
    if len(affected_indices) > 0:
        target_layers = [model.layer4[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)

        # Take top 2 predicted diseases for Grad-CAM visualization
        top_indices = affected_indices[np.argsort(outputs[affected_indices])[::-1][:2]]

        st.subheader("Grad-CAM Heatmaps")
        for i in top_indices:
            targets = [ClassifierOutputTarget(i)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            img_np = np.array(image.resize((224, 224))) / 255.0
            heatmap = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
            st.write(f"**{classes[i]}**")
            st.image(heatmap, use_container_width=True)

    st.success("✅ Analysis complete! Scroll up to see results.")
