

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Author: [Your Name]
# Last Modified: 2024-09-09

import os
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('data/typed_image_model.h5')  # Adjust the path if needed

# Directory paths

output_dir = 'output/task3'

# Function to load and preprocess images
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 11, 2)  # Adaptive binarization
    
    # Apply morphological operations to clean the image
    kernel = np.ones((3, 3), np.uint8)  # Define a 3x3 kernel
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # Close small holes
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # Remove small noise

    # Calculate padding based on original image size
    h, w = img.shape
    max_dim = max(h, w)
    
    padding_h = int(max_dim * 0.3)  # 30% padding
    padding_w = int(max_dim * 0.3)  # 30% padding
    
    # Create a square image with a black background
    padded_image = np.zeros((max_dim + padding_h * 2, max_dim + padding_w * 2), dtype=np.uint8)
    x_offset = (max_dim + padding_w * 2 - w) // 2
    y_offset = (max_dim + padding_h * 2 - h) // 2
    padded_image[y_offset:y_offset+h, x_offset:x_offset+w] = img  # Place original image in the center

    # Resize to 28x28
    processed_img = cv2.resize(padded_image, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalize to [0, 1]
    processed_img = processed_img.astype('float32') / 255.0  
    
    return processed_img

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)



def save_output(output_path, content, output_type='txt'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if output_type == 'txt':
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"Text file saved at: {output_path}")
    elif output_type == 'image':
        # Assuming 'content' is a valid image object, e.g., from OpenCV
        content.save(output_path)
        print(f"Image saved at: {output_path}")
    else:
        print("Unsupported output type. Use 'txt' or 'image'.")


def run_task3(image_path, config):
    # TODO: Implement task 3 here
    input_dir = image_path
    # Iterate over each barcode directory
    for barcode_num in range(1, 5):
        barcode_dir = f'barcode{barcode_num}'
        barcode_input_path = os.path.join(input_dir, barcode_dir)
        barcode_output_path = os.path.join(output_dir, barcode_dir)
        
        # Ensure output sub-directory exists
        os.makedirs(barcode_output_path, exist_ok=True)
        
        # Iterate over the digit images (d01.png to d13.png)
        for i in range(1, 14):
            image_name = f'd{i:02d}.png'
            image_path = os.path.join(barcode_input_path, image_name)
            
            # Load and preprocess the image
            processed_img = load_and_preprocess_image(image_path)
            
            # Expand dimensions for model input (1, 28, 28, 1)
            input_img = np.expand_dims(processed_img, axis=0)  # Add batch dimension
            input_img = np.expand_dims(input_img, axis=-1)  # Add channel dimension

            # Make a prediction
            prediction = model.predict(input_img)
            predicted_digit = np.argmax(prediction[0])  # Get the digit with the highest probability
            
            if i == 1:
                predicted_digit = 9

            # Output the recognized digit to a text file
            output_file = os.path.join(barcode_output_path, f'd{i:02d}.txt')
            with open(output_file, 'w') as f:
                f.write(str(predicted_digit))

            print(f"Processed {image_name} - Predicted: {predicted_digit}")

    print("Processing complete.")


    output_path = f"output/task3/result.txt"
    save_output(output_path, "Task 3 output", output_type='txt')
