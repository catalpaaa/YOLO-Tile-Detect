import yaml

import cv2
import numpy as np
import onnxruntime as ort

from modules.args import args
from modules.visualization import render_onnx_detections


def preprocess(img, input_width, input_height):
    """
    Preprocess the given image for ONNX model inference.

    Parameters:
    - img (numpy.ndarray): The input image in BGR format.
    - input_width (int): The width the model expects for its input.
    - input_height (int): The height the model expects for its input.

    Returns:
    - numpy.ndarray: The preprocessed image data.
    """

    # Convert the image from BGR (OpenCV default) to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize the image to match the input dimensions expected by the ONNX model
    img = cv2.resize(img, (input_width, input_height))

    # Normalize the pixel values to the range [0, 1] (originally 8 bit int)
    image_data = np.array(img) / 255.0

    # Reorder dimensions to make channels the first dimension (required for ONNX)
    image_data = np.transpose(image_data, (2, 0, 1))

    # Add an additional batch dimension
    return np.expand_dims(image_data, axis=0).astype(np.float32)


def postprocess(img, output, input_width, input_height, confidence_thres, iou_thres, tile_text_name):
    """
    Extract bounding boxes, scores, and class IDs from the model's output and 
    render the detections on the input image.

    Args:
    - img (numpy.ndarray): Original input image in BGR format.
    - output (numpy.ndarray): Raw output from the ONNX model.
    - input_width (int): Width of the model input.
    - input_height (int): Height of the model input.
    - confidence_thres (float): Threshold for filtering out low-confidence detections.
    - iou_thres (float): Threshold for non-maximum suppression.
    - tile_text_name (str): Path to save the detection results as a text file.

    Returns:
    - numpy.ndarray: Input image annotated with bounding boxes and class labels.
    """

    # Load class names from the provided YAML file
    with open(args.class_yaml, "r") as file:
        classes = yaml.safe_load(file)["names"]

    # Generate random colors for visualizing different classes
    color_palette = np.random.uniform(0, 255, size=(len(classes), 3))

    # Process the raw output to get a more interpretable format
    outputs = np.transpose(np.squeeze(output[0]))

    # Lists to hold the details of detected objects
    boxes = []
    scores = []
    class_ids = []

    # Calculate the scaling factors to map bounding box coordinates back to the original image size
    x_factor = img.shape[1] / input_width
    y_factor = img.shape[0] / input_height

    # Process each detection from the model's output
    with open(tile_text_name, 'w') as txt_file:
        for i in range(outputs.shape[0]):
            # Extract class scores and find the class with the maximum score
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)

            # Filter out detections with confidence below the threshold
            if max_score >= confidence_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Convert coordinates from relative values to actual pixel values
                left = (x - w / 2) * x_factor
                top = (y - h / 2) * y_factor
                width = w * x_factor
                height = h * y_factor

                # Save detection details in the text file
                txt_file.write(
                    f"{class_id} {left/img.shape[1]:.5f} {top/img.shape[0]:.5f} {width/img.shape[1]:.5f} {height/img.shape[0]:.5f}\n")

                # Append detection details to lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Filter out overlapping bounding boxes using non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)

        # Draw bounding boxes and labels on the original image
        for index in indices:
            render_onnx_detections(
                img, boxes[index], scores[index], class_ids[index], classes, color_palette)

    return img
