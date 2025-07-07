import streamlit as st
import cv2
from PIL import Image
import numpy as np
import Dehaze
import torch
import torchvision.transforms as transforms

# Function to load the PyTorch model
def load_model():
    model = torch.load(r"C:\Users\chyav\Downloads\VIT_Transformer-20241217T062035Z-001\VIT_Transformer\model.pt") 
    return model

# Function to preprocess the image
def preprocess_image(image):
    resized_image = cv2.resize(image, (224, 224))
    tensor_image = transforms.ToTensor()(resized_image)
    normalized_image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor_image)
    preprocessed_image = normalized_image.unsqueeze(0)
    return preprocessed_image

def dehaze_image(model, preprocessed_image):
    output_image = Dehaze.dhazei(preprocessed_image, 0)  
    return output_image

def main():
    st.set_page_config(page_title="VIT Transformers Dehaze App", layout="wide")

    st.title('VIT Transformers Dehaze App')
    st.markdown("---")

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Display image and dehazed image side by side
    col1, col2 = st.columns(2)

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        col1.header("Original Image")
        col1.image(image, use_column_width=True)
        
        if st.button('Dehaze'):
            with st.spinner('Dehazing...'):
                # Load model
                model = load_model()
                img_array = np.array(image)
                preprocessed_image = preprocess_image(img_array)

                # Dehaze the image
                dehazed_img = dehaze_image(model, image)

                # Display dehazed image
                dehazed_pil_img = Image.fromarray(dehazed_img)
                col2.header("Dehazed Image")
                col2.image(dehazed_pil_img, use_column_width=True)
                st.success('Dehazing completed!')
                
if __name__ == '__main__':
    main()