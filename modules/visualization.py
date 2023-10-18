import yaml
import cv2
import numpy as np
from modules.args import args


def render_onnx_detections(image, box, score, class_id, classes, color_palette):
    """
    Draw bounding boxes and class labels on the input image based on the detected objects.

    Args:
        img (numpy.ndarray): The input image on which detections will be drawn.
        box (list): Detected bounding box coordinates as [x, y, width, height].
        score (float): Detection confidence score.
        class_id (int): Class ID of the detected object.
        classes (list): List of all available class names.
        color_palette (list): List of RGB colors corresponding to each class.

    Returns:
        None: This function modifies the input image in-place.
    """

    # Extract coordinates and dimensions of the bounding box
    x1, y1, w, h = map(int, box)

    # Get the corresponding color for the detected class ID
    color = tuple(map(int, color_palette[class_id]))

    # Draw the bounding box rectangle on the image
    cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), color, 2)

    # Construct the label string with class name and confidence score
    label = f'{classes[class_id]}: {score:.2f}'

    # Calculate the size of the label for positioning purposes
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

    # Determine the position for the label
    label_x = x1
    label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + label_size[1]

    # Draw a filled background rectangle for the label text for better visibility
    cv2.rectangle(image, (label_x, label_y - label_size[1]),
                  (label_x + label_size[0], label_y), color, cv2.FILLED)

    # Draw the label text on top of the background rectangle
    cv2.putText(image, label, (label_x, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def render_final_detections(combined_bbox_text_path, original_image_path, annotated_image_path, class_yaml_path):
    """
    Visualize and overlay bounding box detections on the original image.

    Parameters:
    - combined_bbox_text_path (str): Path to the file containing bounding box information.
    - original_image_path (str): Path to the original image where detections will be overlaid.
    - annotated_image_path (str): Output path to save the image with overlaid detections.
    - class_yaml_path (str): Path to the YAML file that contains class names.

    Returns:
    None: This function saves the annotated image to the specified path.
    """

    # Load class names from the provided YAML file
    with open(class_yaml_path, "r") as file:
        class_names = yaml.safe_load(file)["names"]

    # Create a color palette to visually differentiate each class
    color_palette = np.random.uniform(0, 255, size=(len(class_names), 3))

    # Read and parse detected bounding boxes from the provided text file
    with open(combined_bbox_text_path, "r") as file:
        bounding_boxes = [line.strip().split() for line in file.readlines()]

    # Load the original image to be annotated
    image = cv2.imread(original_image_path)

    # Extract the dimensions of the image for coordinate conversion
    height, width, _ = image.shape

    # Overlay each detected bounding box and its corresponding label on the image
    for bbox in bounding_boxes:
        class_index, x, y, w, h = map(float, bbox)
        class_id = int(class_index)

        # Convert relative bounding box coordinates to absolute pixel values
        if args.use_onnx:
            x1 = int(x * width)
            y1 = int(y * height)
        else:
            x1 = int((x - w / 2) * width)
            y1 = int((y - h / 2) * height)
        w = int(w * width)
        h = int(h * height)

        # Get the RGB color associated with the class ID
        color = tuple(map(int, color_palette[class_id]))

        # Draw the detected bounding box on the image
        cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), color, 2)

        # Create the label containing the class name
        label = f"{class_names[class_id]}"

        # Determine the size of the label text for positioning
        label_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

        # Define the position for the label and draw a background rectangle for better readability
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + label_size[1]
        cv2.rectangle(image, (label_x, label_y -
                      label_size[1]), (label_x + label_size[0], label_y + 5), color, cv2.FILLED)

        # Overlay the label text on the image
        cv2.putText(image, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Save the annotated image with detections to the specified path
    cv2.imwrite(annotated_image_path, image)
