

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

import cv2
import numpy as np


# Function to display images using matplotlib
def display_image(img, title=""):
    plt.figure(figsize=(5, 5))
    if len(img.shape) == 2:  # Grayscale image
        plt.imshow(img, cmap='gray')
    else:  # Color image
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Function to show connected components with different colors
def imshow_components(labels):
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0
    return labeled_img

# Function to extract features from each blob
def extract_features(labels_im, stats):
    features = []
    for i, stat in enumerate(stats):
        x, y, width, height, area = stat
        blob = labels_im[y:y+height, x:x+width] == 1
        features.append({
            "Area": area,
            "Height": height,
            "Width": width,
            "Fraction of Foreground Pixels": area / (height * width),
            "Distribution in X-direction": np.sum(blob, axis=0),
            "Distribution in Y-direction": np.sum(blob, axis=1)
        })
    return features

def reverse_transform(points, rotation_matrix, crop_offset):
    # Undo cropping: Add cropping offset back to the points
    points_with_crop_offset = points + crop_offset

    # Convert points to homogeneous coordinates for matrix multiplication
    points_homogeneous = np.hstack([points_with_crop_offset, np.ones((points_with_crop_offset.shape[0], 1))])

    # Calculate the inverse of the rotation matrix
    inverse_rotation_matrix = cv2.invertAffineTransform(rotation_matrix)

    # Apply the inverse rotation matrix to the points
    original_points = (inverse_rotation_matrix @ points_homogeneous.T).T

    # Return the transformed points (now in original image coordinates)
    return original_points[:, :2]  # Only take the x, y coordinates


def display_original_image_with_points(original_image, points):
    # Draw the points on the original image
    for point in points:
        cv2.circle(original_image, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)  # Green circles for points

    # Display the image
    cv2.imshow(original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Function to save coordinates to a text file
def save_coordinates(file_name, points):
    with open(file_name, 'w') as file:
        file.write(', '.join([f"{int(p[0])}, {int(p[1])}" for p in points]))


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


def run_task1(image_path, config):
    # TODO: Implement task 1 here

    output_folder = "task4_mods/task4_mod_out/task1"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize the barcode detector
    barcode_detector = cv2.barcode_BarcodeDetector()


    # Process each image from img1 to img5
    for i in range(1, 6):
        # Read the image containing a barcode
        single_image_path = f'{image_path}/img{i}.jpg'  # Construct the image filename
        image = cv2.imread(single_image_path)

        # Check if the image was loaded successfully
        if image is None:
            print(f"Error: Could not read image {single_image_path}")
            continue

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to the equalized image
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Apply a mean filter (linear filtering)
        mean_filter = np.ones((3, 3), np.float32) / 9
        filtered_image = cv2.filter2D(blurred_image, -1, mean_filter)

        # Apply thresholding to create a binary image
        _, thresholded_image = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Detect the barcode and get its bounding box (without decoding)
        ok, points = barcode_detector.detect(thresholded_image)

        if ok:
            # Convert points to integer format for drawing
            points = points.astype(int)

            # Calculate the angle of rotation and the center of the bounding shape
            rect = cv2.minAreaRect(points[0])  # Get the minimum area rectangle
            center = (int(rect[0][0]), int(rect[0][1]))

            # Calculate the angle and the dimensions of the rectangle
            angle = rect[-1]  # The angle is the last element of the rect tuple
            width, height = rect[1]  # Width and height of the bounding box

            # Determine the correct angle for rotation
            if width < height:
                angle += 90  # Rotate to make the longer side horizontal

            # Prepare for image rotation
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            #rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

            (h, w) = image.shape[:2]
            (cX, cY) = (w // 2, h // 2)  # Center of the image

            # Compute the new bounding dimensions
            new_width = int(abs(w * np.cos(np.radians(angle))) + abs(h * np.sin(np.radians(angle))))
            new_height = int(abs(h * np.cos(np.radians(angle))) + abs(w * np.sin(np.radians(angle))))

            # Update the transformation matrix with the new center
            M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
            M[0, 2] += (new_width / 2) - cX
            M[1, 2] += (new_height / 2) - cY

            # Rotate the image
            rotated_image = cv2.warpAffine(image, M, (new_width, new_height))


            # Calculate the bounding box
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Transform bounding box points to match the rotated image
            transformed_box = []
            for point in box:
                # Apply the rotation transformation to each point
                transformed_point = M @ np.array([point[0], point[1], 1])  # Homogeneous coordinates
                transformed_box.append(transformed_point[:2])  # Keep only x and y
            transformed_box = np.int0(transformed_box)

            # Calculate dynamic padding based on the height of the bounding box
            padding = int(width * 0.1)  # Adjust padding as needed

            # Create a new padded box around the original box
            padded_box = np.array([
                [min(transformed_box[:, 0]) - padding, min(transformed_box[:, 1]) - padding],
                [max(transformed_box[:, 0]) + padding, min(transformed_box[:, 1]) - padding],
                [max(transformed_box[:, 0]) + padding, max(transformed_box[:, 1]) + padding * 2],
                [min(transformed_box[:, 0]) - padding, max(transformed_box[:, 1]) + padding * 2]
            ])

            # Crop the region of interest (ROI) using the padded box
            x_min, y_min = np.min(padded_box[:, 0]), np.min(padded_box[:, 1])
            x_max, y_max = np.max(padded_box[:, 0]), np.max(padded_box[:, 1])

            # Ensure cropping limits are within the image boundaries
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(rotated_image.shape[1], x_max)
            y_max = min(rotated_image.shape[0], y_max)

            # Crop the padded region from the rotated image
            cropped_image = rotated_image[y_min:y_max, x_min:x_max]


            # Optionally, display the cropped image
            #cv2.imshow(cropped_image)
            #cv2.waitKey(0)








            print(f"Processing img{i}.jpg")
            cimg = cropped_image

            # Convert to gray and apply Gaussian blur
            gray = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # Thresholding using Otsu's method
            _, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


            # Invert the image for connected component analysis
            th = cv2.bitwise_not(th3)

            # Connected component analysis
            num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(th, connectivity=8)


            # Extract features for each blob
            blob_features = extract_features(labels_im, stats)

            # Compute the image area
            image_area = cimg.shape[0] * cimg.shape[1]

            # Set relative area thresholds (percentage of the total image area)
            min_area_percentage = 0.0005  # 0.1% of the total image area
            max_area_percentage = 0.1    # 10% of the total image area

            # Calculate the actual area thresholds
            min_area = min_area_percentage * image_area
            max_area = max_area_percentage * image_area

            min_aspect_ratio = 0.3  # Minimum aspect ratio (width/height)
            max_aspect_ratio = 1.8  # Maximum aspect ratio (to filter lines)

            # Variables to store the leftmost and rightmost blobs
            leftmost_x = None
            rightmost_x = None
            leftmost_blob = None
            rightmost_blob = None

            # Filter and analyze blobs based on relative area and aspect ratio
            for k in range(1, num_labels):  # Start from 1 to exclude the background
                x, y, w, h, area = stats[k]

                # Compute aspect ratio
                aspect_ratio = float(w) / h

                # Filter based on area and aspect ratio
                if min_area <= area <= max_area and min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                    # Extract the character patch
                    character_patch = gray[y:y+h, x:x+w]



                    # Draw bounding box on the original image
                    output = cimg.copy()
                    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)


                    # Track leftmost and rightmost blobs
                    if leftmost_x is None or x < leftmost_x:
                        leftmost_x = x
                        leftmost_blob = (x, y, w, h)

                    if rightmost_x is None or x + w > rightmost_x:
                        rightmost_x = x + w
                        rightmost_blob = (x, y, w, h)

            # Padding factor (e.g., 5% of the parallelogram height)
            padding_factor = 0.1

            # Draw a parallelogram enclosing the leftmost and rightmost blobs
            if leftmost_blob is not None and rightmost_blob is not None:
                lx, ly, lw, lh = leftmost_blob
                rx, ry, rw, rh = rightmost_blob

                # Define the corners of the parallelogram
                top_left = (lx, ly)
                top_right = (rx + rw, ry)
                bottom_left = (lx, ly + lh)
                bottom_right = (rx + rw, ry + rh)

                # Calculate the bounding box dimensions for the entire parallelogram
                total_width = top_right[0] - top_left[0]
                total_height = bottom_left[1] - top_left[1]

                # Calculate padding (e.g., 5% of the height)
                padding = int(total_height * padding_factor)

                # Adjust corners with padding (make sure we stay within image bounds)
                top_left_padded = (max(top_left[0] - padding, 0), max(top_left[1] - padding, 0))
                top_right_padded = (min(top_right[0] + padding, cimg.shape[1]), max(top_right[1] - padding, 0))
                bottom_left_padded = (max(bottom_left[0] - padding, 0), min(bottom_left[1] + padding, cimg.shape[0]))
                bottom_right_padded = (min(bottom_right[0] + padding, cimg.shape[1]), min(bottom_right[1] + padding, cimg.shape[0]))

                # Define the points for the padded parallelogram
                padded_parallelogram_points = np.array([top_left_padded, top_right_padded, bottom_right_padded, bottom_left_padded], np.float32)

                # Calculate the center of the padded parallelogram
                center = np.mean(padded_parallelogram_points, axis=0)

                # Define a scale factor to enlarge the parallelogram
                scale_factor = 1.0  # Adjust this value to increase size (1.0 means no change)

                # Scale the points outward from the center
                scaled_points = []
                for point in padded_parallelogram_points:
                    scaled_point = center + (point - center) * scale_factor
                    scaled_points.append(scaled_point)

                scaled_parallelogram_points = np.array(scaled_points, np.float32)

                # Define the target rectangle dimensions
                width = int(np.linalg.norm(np.array(scaled_parallelogram_points[0]) - np.array(scaled_parallelogram_points[1])))
                height = int(np.linalg.norm(np.array(scaled_parallelogram_points[0]) - np.array(scaled_parallelogram_points[3])))

                # Define the target rectangle points
                target_rectangle = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)

                # Get the perspective transform matrix
                matrix = cv2.getPerspectiveTransform(scaled_parallelogram_points, target_rectangle)

                # Apply the perspective transformation
                warped_image = cv2.warpPerspective(cimg, matrix, (width, height))

                # Create a black background
                black_background = np.zeros((height, width, 3), dtype=np.uint8)

                # Place the cropped parallelogram on the black background
                final_output = cv2.add(black_background, warped_image)

                # Offset used when cropping the image
                crop_offset = np.array([x_min, y_min])

                # Get the original coordinates on the unprocessed input image
                original_points = reverse_transform(padded_parallelogram_points, M, crop_offset)


                # Display the original image with the transformed points
                #display_original_image_with_points(image, original_points)


                # Display the cropped and padded parallelogram on black background
                #display_image(final_output, title="Cropped and Padded Parallelogram on Black Background")

                # Draw the padded parallelogram on the original image for reference
                #cv2.polylines(output, [scaled_parallelogram_points.astype(np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
                #display_image(output, title="Padded Parallelogram Enclosing Leftmost and Rightmost Blobs")


                # Save the cropped barcode image
                barcode_image_path = f'{output_folder}/barcode{i}.png'
                cv2.imwrite(barcode_image_path, final_output)

                # Save original coordinates to a text file
                coordinates_file = f'{output_folder}/img{i}.txt'
                save_coordinates(coordinates_file, original_points)

                print(f"Saved barcode image as {barcode_image_path} and coordinates in {coordinates_file}")



        else:
            print(f"Barcode not detected in image {single_image_path}")



    output_path = f"output/task1/result.txt"
    save_output(output_path, "Task 1 output", output_type='txt')
