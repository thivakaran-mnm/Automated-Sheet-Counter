# Automated Sheet Counter using Deep Learning

This project involves creating an automated sheet counter using deep learning techniques. The application is built using Streamlit for the frontend and TensorFlow/Keras for the machine learning model. The system is designed to count the number of sheets in an uploaded image.

## Project Components

1. **Jupyter Notebook**: Contains the code for training the deep learning model.
2. **Streamlit Application (`sheet_count_app.py`)**: Provides the user interface for interacting with the model.

## Prerequisites

Ensure you have the following installed:

- Python 3.8 or higher
- TensorFlow
- OpenCV
- Streamlit
- Numpy

You can install the required packages using pip:

pip install tensorflow opencv-python-headless streamlit numpy 

## Project Structure
- sheet_count_app.py: Streamlit application for sheet counting.
- sheet_count_model_1.keras/: Directory where the trained model file should be stored.
- automated_sheet_counter.ipyn/: Directory containing Jupyter Notebooks for model training.

## Setting Up the Environment

1. Clone the Repository

   If you haven’t already, clone the repository:
   git clone <repository-url>
   cd <repository-folder>

2. Place the Trained Model

   Ensure you have the trained model file saved as sheet_count_model_1.keras. Place this file in the trained_model/ directory. If the 
   directory doesn’t exist, create it:

   mkdir trained_model

   Move your model file into this directory.

## Running the Streamlit Application
1. Update Model Path

   Open sheet_count_app.py and ensure the path to the trained model is correctly specified:

   model = load_model('trained_model/sheet_count_model_1.keras')

   If your model is in a different directory, update the path accordingly.

2. Run the Application

   Launch the Streamlit application by running:

   streamlit run app.py

   This command will open the Streamlit interface in your default web browser.

## Using the Application
1. Upload an Image

   - Navigate to the Streamlit interface in your browser.
   - Use the file uploader widget to select and upload an image file (supported formats: jpg, jpeg, png).
2. View Predictions

   - The application will process the uploaded image, make predictions using the pre-trained model, and display the estimated number of sheets.

## Jupyter Notebook for Model Training 
If you need to retrain the model or view the training process, open the Jupyter Notebook files in the notebooks/ directory. Ensure that the notebook is configured with the correct paths and dependencies.

## Troubleshooting
- Model Not Loading: Verify that the model file path is correct and that the model file is properly placed in the sheet_count_model_/ directory.
- Image Processing Errors: Ensure the uploaded image is in the supported format and is not corrupted.
- Dependencies Issues: Check that all required Python packages are installed. You can use pip freeze to verify the installed packages.

## Future Improvements
- Enhance image preprocessing techniques for better accuracy.
- Implement additional features such as batch processing and more robust error handling.

## Contact
For any questions or issues, please contact [thivakaran.mnm@gmail.com].
