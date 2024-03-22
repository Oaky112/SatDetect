import streamlit as st  # type: ignore
import cv2
from ultralytics import YOLO
import requests  # type: ignore
from PIL import Image
import os
import io

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Function to load the YOLO model
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model

# Set confidence and IOU thresholds
conf_threshold = 0.20  # Static confidence threshold
iou_threshold = 0.5    # Static IOU threshold

# Function to predict objects in the image
def predict_image(model, image, conf_threshold, iou_threshold):
    # Predict objects using the model
    res = model.predict(
        image,
        conf=conf_threshold,
        iou=iou_threshold,
        device='cpu',
    )
    
    class_name = model.model.names
    classes = res[0].boxes.cls
    class_counts = {}
    
    # Count the number of occurrences for each class
    for c in classes:
        c = int(c)
        class_counts[class_name[c]] = class_counts.get(class_name[c], 0) + 1

    # Generate prediction text
    prediction_text = 'Predicted '
    for k, v in sorted(class_counts.items(), key=lambda item: item[1], reverse=True):
        prediction_text += f'{v} {k}'
        
        if v > 1:
            prediction_text += 's'
        
        prediction_text += ', '

    prediction_text = prediction_text[:-2]
    if len(class_counts) == 0:
        prediction_text = "No objects detected"

    # Calculate inference latency
    latency = sum(res[0].speed.values())  # in ms, need to convert to seconds
    latency = round(latency / 1000, 2)
    prediction_text += f' in {latency} seconds.'

    # Convert the result image to RGB
    res_image = res[0].plot()
    res_image = cv2.cvtColor(res_image, cv2.COLOR_BGR2RGB)
    
    return res_image, prediction_text

def main():
    # Set Streamlit page configuration
    st.set_page_config(
        page_title="Satellite Wildfire Detection App",
        page_icon="🌎",
        initial_sidebar_state="collapsed",
    )

    # Custom CSS and Background Image
    custom_css = """
    <style>
    /* Overall App Styling */
    body {
        font-family: 'Arial', sans-serif;
        background-color: #ecf0f5;  /* Light blue with clouds background */
        color: #333333;  /* Dark grey text color */
        line-height: 1.6;
    }

    /* Page Container */
    .container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }

    /* Title Styling */
    .title {
        text-align: center;
        font-size: 75px;  /* Bigger font size */
        font-weight: bold;
        margin-bottom: 20px;
        color: #ffffff;  /* White color */
    }

    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background-color: #ffffff;  /* White sidebar background */
        box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);  /* Sidebar shadow effect */
        padding: 20px;
        border-radius: 5px;
    }
    .sidebar .sidebar-content .sidebar-close-button {
        color: #666666;  /* Grey color for sidebar close button */
    }

    /* Form and Input Styling */
    .stRadio .label {
        font-size: 20px;
        color: #333333;  /* Dark grey color */
    }
    .stTextInput input {
        border-radius: 5px;
        border: 1px solid #cccccc;  /* Light grey border */
    }

    /* Button Styling */
    .stButton button {
        background-color: #4fb2cf;  /* Light blue button background */
        color: #ffffff;  /* White text color */
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #3a99b5;  /* Slightly darker blue on hover */
    }

    /* Image Styling */
    img {
        max-width: 100%;
        height: auto;
    }

    /* Error Message Styling */
    .error-message {
        color: #ff0000;  /* Red error text color */
    }

    /* Background Image */
    .stApp {
        margin: auto;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        overflow: auto;
        background-image: url('https://mrwallpaper.com/images/hd/download-firewatch-wallpaper-ervaypm7906b8op2.jpg'); 
        background-size: 160% auto; /* Change size as needed *
        background-repeat: no-repeat; /* Adjust as needed */
        background-position: center 140%; /* Adjust as needed */
    }
    </style>
    """

    st.markdown(custom_css, unsafe_allow_html=True)

    # App title
    st.markdown("<div class='title'>Satellite Wildfire Detection</div>", unsafe_allow_html=True)
    

    st.markdown(
            """
            <div style='text-align: center;'>
                <center>
                <h3><strong>Enhancing wildfire prediction and management through real-time computer vision</strong></h3>
                </center>
            </div>

            <div style='text-align: justify;'>
                <h3 style="text-decoration: underline;">My Mission and How it Works</h3>
                <p>Developed for my 4th year honors project.</p>
                <p>The app utilizes two key models:</p>
                <center>
                    <ul>
                        <li><strong>Fire-Detection Model:</strong> Trained using the D-Fire dataset, this model detects signs of fire and smoke including my own created dataset of clouds and wildfires in satellite images.</li>
                        <li><strong>YOLOv8 Model:</strong> A general-purpose object detection model used to identify various objects relevant to wildfire monitoring.</li>
                    </ul>
                </center>
                <p>Images for training were obtained by scraping Google Chrome and picking through the images that are relevant then annotation them on cvat, and the YOLOv8 model used is developed by Ultralytics, known for high-performance deep learning models.</p>
            </div>
            """,
            unsafe_allow_html=True
        )


    # Model selection
    model_type = st.radio("Select Model Type", ("Fire Detection", "General"), index=0)

    models_dir = "general-models" if model_type == "General" else "fire-models"
    model_files = [f.replace(".pt", "") for f in os.listdir(models_dir) if f.endswith(".pt")]
    
    selected_model = st.selectbox("Select Model Size", sorted(model_files), index=2)

    # Load the selected model
    model_path = os.path.join(models_dir, selected_model + ".pt") #type: ignore
    model = load_model(model_path)

    # Image selection
    image = None
    image_source = st.radio("Select image source:", ("Enter URL", "Upload from Computer"))
    if image_source == "Upload from Computer":
        # File uploader for image
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
        else:
            image = None

    else:
        # Input box for image URL
        url = st.text_input("Enter the image URL:")
        if url:
            try:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    image = Image.open(response.raw)
                else:
                    st.error("Error loading image from URL.")
                    image = None
            except requests.exceptions.RequestException as e:
                st.error(f"Error loading image from URL: {e}")
                image = None

    if image:
        # Display the uploaded image
        with st.spinner("Detecting"):
            prediction, text = predict_image(model, image, conf_threshold, iou_threshold)
            st.image(prediction, caption="Prediction", use_column_width=True)
            st.success(text)
            
        prediction = Image.fromarray(prediction)

        # Create a BytesIO object to temporarily store the image data
        image_buffer = io.BytesIO()

        # Save the image to the BytesIO object in PNG format
        prediction.save(image_buffer, format='PNG')

        # Create a download button for the image
        st.download_button(
            label='Download Prediction',
            data=image_buffer.getvalue(),
            file_name='prediction.png',
            mime='image/png'
        )

if __name__ == "__main__":
    main()

        
       
