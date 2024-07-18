import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image, ImageDraw

# Load the saved model
model = joblib.load('trained_model.pkl')

# Define the classes
classes = {0: 'No Tumor', 1: 'Positive Tumor'}

def preprocess_image(image):
    # Preprocess the image as required (resize, normalize, etc.)
    processed_image = cv2.resize(image, (200, 200))
    processed_image = processed_image.reshape(1, -1) / 255.0
    return processed_image

def predict(image):
    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make predictions using the loaded model
    prediction = model.predict(processed_image)

    # Map the predicted class to its corresponding label
    predicted_class = classes[prediction[0]]

    return predicted_class

def main():
    # Set page configuration
    st.set_page_config(
        page_title='Brain Tumor Classification',
        page_icon='🧠',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    # CSS styling with background image
    st.markdown(
       """
        <style>
        .stApp {
           background-image: url('https://www.google.com/imgres?q=black%20screen&imgurl=https%3A%2F%2Fwww.ledr.com%2Fcolours%2Fblack.jpg&imgrefurl=https%3A%2F%2Fwww.ledr.com%2Fcolours%2Fblack.htm&docid=o49zPtLuSQ5KUM&tbnid=h2wDr2_G8qUYoM&vet=12ahUKEwi3rP2gv-yFAxWAb2wGHfWoBJkQM3oECBUQAA..i&w=1278&h=990&hcb=2&ved=2ahUKEwi3rP2gv-yFAxWAb2wGHfWoBJkQM3oECBUQAA');
           background-size: cover;
           background-repeat: no-repeat;
           background-attachment: fixed;
        }
        .header {
            padding: 20px;
            text-align: center;
        }
        .upload-section {
            margin-top: 40px;
            text-align: center;
        }
        .result-section {
            margin-top: 40px;
            padding: 20px;
            border: 2px solid #e0e0e0;
            background-color: #ffffff;
            text-align: center;
        }
        .options-section {
            margin-top: 40px;
            padding: 20px;
            border: 2px solid #e0e0e0;
            background-color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Header section
    st.title('Brain Tumor Classification')
    st.markdown("Upload an MRI brain image and get the tumor classification result.")

    # File upload section
    st.subheader('Upload an MRI brain image')
    uploaded_file = st.file_uploader('', type=['jpg', 'jpeg', 'png'], help='Supported formats: JPG, JPEG, PNG')

    if uploaded_file is not None:
        # Open the image file using PIL
        image = Image.open(uploaded_file)
        # Convert the image to grayscale
        image_gray = image.convert('L')
        # Convert the PIL image to NumPy array
        image_array = np.array(image_gray)

        # Display the uploaded image
        st.image(image_array, caption='Uploaded MRI Image', use_column_width=True)

        # Perform prediction
        result = predict(image_array)

        # Display the predicted result
        st.subheader('Prediction')
        st.markdown(f"*Result*: {result}", unsafe_allow_html=True)
        if result == 'Positive Tumor':
            st.error('Tumor detected')
        else:
            st.success('No tumor detected')

    # Additional options section
    st.subheader('Additional Options')
    option = st.selectbox('Select an option', ['None', 'Processed Image with Tumor Highlighted', 'Image Histogram', 'Image Statistics'])

    if option == 'Processed Image with Tumor Highlighted':
        # Display processed image with tumor highlighted
        if uploaded_file is not None:
            processed_image = preprocess_image(image_array)
            processed_image_copy = processed_image.reshape((200, 200))

            # Draw a circle around the tumor region on the processed image
            if result == 'Positive Tumor':
                # Detect tumor location (dummy example, you should replace this with your own tumor detection algorithm)
                tumor_location = (100, 100)
                radius = 50

                # Create a copy of the processed image and draw a circle
                processed_image_copy_rgb = cv2.cvtColor((processed_image_copy * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                image_pil = Image.fromarray(processed_image_copy_rgb)
                draw = ImageDraw.Draw(image_pil)
                draw.ellipse((tumor_location[0] - radius, tumor_location[1] - radius,
                              tumor_location[0] + radius, tumor_location[1] + radius), outline='red', width=2)
                processed_image_copy_rgb = np.array(image_pil)

                # Display the original image and the processed image side by side
                col1, col2 = st.columns(2)
                col1.subheader('Original Image')
                col1.image(image, use_column_width=True)
                col2.subheader('Processed Image with Tumor Highlighted')
                col2.image(processed_image_copy_rgb, caption='Preprocessed Image', use_column_width=True)
            else:
                # Display only the processed image
                st.subheader('Processed Image')
                st.image((processed_image_copy * 255).astype(np.uint8), caption='Preprocessed Image', use_column_width=True)
    elif option == 'Image Histogram':
        # Show image histogram
        if uploaded_file is not None:
            st.subheader('Image Histogram')
            histogram = cv2.calcHist([image_array], [0], None, [256], [0, 256])
            st.bar_chart(histogram)
    elif option == 'Image Statistics':
        # Show image statistics
        if uploaded_file is not None:
            st.subheader('Image Statistics')
            stats_container = st.container()
            with stats_container:
                col1, col2 = st.columns(2)
                col1.write('Min Pixel Value:')
                col1.write(np.min(image_array))
                col2.write('Max Pixel Value:')
                col2.write(np.max(image_array))

# Run the app
if __name__ == '__main__':
    main()