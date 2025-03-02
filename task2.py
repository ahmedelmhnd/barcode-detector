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

import cv2
import numpy as np
import os
from PIL import Image

def save_output(output_path, content, output_type='txt'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if output_type == 'txt':
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"Text file saved at: {output_path}")
    elif output_type == 'image':
        # Convert NumPy array to a Pillow Image
        image = Image.fromarray(content)
        image.save(output_path)
        print(f"Image saved at: {output_path}")
    else:
        print("Unsupported output type. Use 'txt' or 'image'.")

# Analyze connected components for an image
def analyze_image(img, output_dir, img_name, min_area_ratio, max_area_ratio, min_aspect_ratio, max_aspect_ratio, padding_area_ratio=0.001):
    # Convert to gray and apply Gaussian blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Otsu's thresholding
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert the image for connected component analysis
    th = cv2.bitwise_not(th)

    # Connected component analysis
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(th, connectivity=8)

    # Get the total image dimensions and area
    image_height, image_width = gray.shape
    image_area = image_width * image_height

    # Calculate padding based on total image area
    padding = int(np.sqrt(image_area * padding_area_ratio))

    digit_data = []  # List to hold (x, y, w, h, area) tuples for sorting

    # Analyze each component
    for k in range(1, num_labels):  # Start from 1 to exclude the background
        x, y, w, h, area = stats[k]

        # Compute aspect ratio
        aspect_ratio = float(w) / h

        # Normalize the area relative to the total image area
        normalized_area = area / image_area

        # Filter based on relative area and aspect ratio
        if min_area_ratio <= normalized_area <= max_area_ratio and min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
            digit_data.append((x, y, w, h, area))  # Append data for sorting

    # Sort the digits based on their x coordinate (left to right)
    digit_data.sort(key=lambda item: item[0])  # Sort by x coordinate

    # Create output files for each digit
    for digit_count, (x, y, w, h, area) in enumerate(digit_data, start=1):
        # Apply consistent padding to the bounding box coordinates based on image area
        x_padded = max(x - padding, 0)
        y_padded = max(y - padding, 0)
        w_padded = min(w + 2 * padding, image_width - x_padded)  # Ensure we stay within image bounds
        h_padded = min(h + 2 * padding, image_height - y_padded)

        # Extract the character patch from the grayscale image
        character_patch = gray[y_padded:y_padded + h_padded, x_padded:x_padded + w_padded]

        # Save the cropped digit as an image using save_output function
        digit_filename = f"d{digit_count:02d}.png"  # Change made here
        save_output(os.path.join(output_dir, digit_filename), character_patch, output_type='image')

        # Save the bounding box coordinates to a text file (with padding)
        coord_filename = f"d{digit_count:02d}.txt"  # Change made here
        # Calculate bottom-right coordinates
        bottom_right_x = x_padded + w_padded
        bottom_right_y = y_padded + h_padded
        # Adjust the output format to x1, y1, x2, y2
        coordinates_content = f"{x_padded} {y_padded} {bottom_right_x} {bottom_right_y}\n"
        save_output(os.path.join(output_dir, coord_filename), coordinates_content, output_type='txt')

    return [(x, y, w, h) for (x, y, w, h, area) in digit_data]  # Return coordinates


# Main function to run task 2 for all images
def run_task2(image_path, config):
    submission_dir = 'output/task2'
    os.makedirs(submission_dir, exist_ok=True)

    # List of image filenames
    image_filenames = [
        "barcode1.png",
        "barcode2.png",
        "barcode3.png",
        "barcode4.png"
    ]

    # Set area and aspect ratio limits relative to image size
    min_area_ratio = 0.001  # Minimum area as a ratio of the total image area
    max_area_ratio = 0.05  # Maximum area as a ratio of the total image area
    min_aspect_ratio = 0.3  # Minimum aspect ratio (width/height) for digits
    max_aspect_ratio = 1.8  # Maximum aspect ratio to exclude lines

    # Process and analyze each image
    for filename in image_filenames:
        img_path = os.path.join(image_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        # Create a directory for each image
        img_name = os.path.splitext(filename)[0]
        output_dir = os.path.join(submission_dir, img_name)
        os.makedirs(output_dir, exist_ok=True)

        # Analyze the image
        analyze_image(img, output_dir, img_name, min_area_ratio, max_area_ratio, min_aspect_ratio, max_aspect_ratio)








