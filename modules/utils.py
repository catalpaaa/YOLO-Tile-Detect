import os
import glob
import concurrent.futures

import torch
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

from modules.args import args


def euclidean_distance(box1, box2):
    """
    Compute the Euclidean distance between the centers of two bounding boxes.

    Parameters:
    - box1 (list): First bounding box as [class_id, x_center, y_center, width, height].
    - box2 (list): Second bounding box as [class_id, x_center, y_center, width, height].

    Returns:
    float: Euclidean distance between the centers of box1 and box2.
    """

    # Extract center coordinates of both boxes
    x1_center, y1_center = box1[1], box1[2]
    x2_center, y2_center = box2[1], box2[2]

    # Calculate and return the Euclidean distance
    return ((x2_center - x1_center) ** 2 + (y2_center - y1_center) ** 2) ** 0.5


def iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Parameters:
    - box1 (list): First bounding box as [class_id, x_center, y_center, width, height].
    - box2 (list): Second bounding box as [class_id, x_center, y_center, width, height].

    Returns:
    float: IoU value between box1 and box2.
    """

    # Extract the necessary parameters from the boxes
    x1_center, y1_center, width1, height1 = box1[1:]
    x2_center, y2_center, width2, height2 = box2[1:]

    # Convert center-coordinates to corner-coordinates for both boxes
    x1_min, y1_min = x1_center - width1 / 2, y1_center - height1 / 2
    x1_max, y1_max = x1_center + width1 / 2, y1_center + height1 / 2
    x2_min, y2_min = x2_center - width2 / 2, y2_center - height2 / 2
    x2_max, y2_max = x2_center + width2 / 2, y2_center + height2 / 2

    # Calculate the area of intersection
    intersection_x_min = max(x1_min, x2_min)
    intersection_y_min = max(y1_min, y2_min)
    intersection_x_max = min(x1_max, x2_max)
    intersection_y_max = min(y1_max, y2_max)
    intersection_area = max(0, intersection_x_max - intersection_x_min) * \
        max(0, intersection_y_max - intersection_y_min)

    # Calculate the area of union
    box1_area = width1 * height1
    box2_area = width2 * height2
    union_area = box1_area + box2_area - intersection_area

    # Calculate and return the IoU
    return intersection_area / union_area if union_area != 0 else 0


def read_tile_detection_output(tile_output_path, tile_width, img_width, tile_height, img_height):
    """
    Extract and adjust the coordinates of detections from a specific tile output based on tile's position.

    Parameters:
    - tile_output_path (str): Path to the tile output file.
    - tile_width (int): Width of each tile.
    - img_width (int): Width of the original image.
    - tile_height (int): Height of each tile.
    - img_height (int): Height of the original image.

    Returns:
    List[List[float]]: Adjusted detections extracted from the tile.
    """

    detections = []

    # Read the tile output file and adjust each detection's coordinates
    with open(tile_output_path, "r") as file:
        for line in file.readlines():
            class_id, x_center, y_center, width, height = map(
                float, line.strip().split())
            tile_x_offset = int(tile_output_path.split("_x")[-1].split(".")[0])
            tile_y_offset = int(
                tile_output_path.split("_y")[-1].split("_x")[0])

            # Adjust bounding box coordinates based on the tile's position in the original image
            x_center = (x_center * tile_width + tile_x_offset) / img_width
            y_center = (y_center * tile_height + tile_y_offset) / img_height
            width = width * tile_width / img_width
            height = height * tile_height / img_height

            detections.append([class_id, x_center, y_center, width, height])

    return detections


def combine_tile_bboxes(temp_output_folder, base_image_name, img_width, img_height, tile_width, tile_height):
    """
    Combine and merge bounding boxes from individual tiles to produce a unified set of detections.

    Parameters:
    - temp_output_folder (str): Directory containing individual tile's detection results.
    - base_image_name (str): Base name of the original image.
    - img_width (int): Width of the original image.
    - img_height (int): Height of the original image.
    - tile_width (int): Width of each tile.
    - tile_height (int): Height of each tile.

    Returns:
    str: Path to the text file containing the combined bounding boxes.
    """

    all_detections = []

    # Concurrently read and process detections from all tile outputs
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(read_tile_detection_output, tile_output_path, tile_width, img_width, tile_height, img_height)
                   for tile_output_path in glob.glob(os.path.join(temp_output_folder, "*.txt"))]

        for future in concurrent.futures.as_completed(futures):
            all_detections.extend(future.result())

    # Merge overlapping detections
    merged_detections = []

    def merge_detections(detection1, all_detections):
        overlapping_detections = [detection1]

        # Check for overlapping detections based on IoU and Euclidean distance thresholds
        for detection2 in all_detections[:]:
            if detection1[0] == detection2[0] and (
                iou(detection1, detection2) > args.iou_threshold
                or euclidean_distance(detection1, detection2) < args.dist_threshold
            ):
                overlapping_detections.append(detection2)
                all_detections.remove(detection2)

        # Merge overlapping detections into a single bounding box
        if len(overlapping_detections) > 1:
            # Calculate merged bounding box parameters
            min_x = min([x_center - width / 2 for _, x_center,
                        _, width, _ in overlapping_detections])
            min_y = min([y_center - height / 2 for _, _, y_center,
                        _, height in overlapping_detections])
            max_x = max([x_center + width / 2 for _, x_center,
                        _, width, _ in overlapping_detections])
            max_y = max([y_center + height / 2 for _, _, y_center,
                        _, height in overlapping_detections])

            merged_x_center = (min_x + max_x) / 2
            merged_y_center = (min_y + max_y) / 2
            merged_width = max_x - min_x
            merged_height = max_y - min_y

            return [detection1[0], merged_x_center, merged_y_center, merged_width, merged_height]
        else:
            return detection1

    # Use ThreadPoolExecutor to merge detections concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(
            merge_detections, detection, all_detections.copy()) for detection in all_detections]
        for future in concurrent.futures.as_completed(futures):
            merged_detections.append(future.result())

    # Save the combined detections to a new file
    merged_output_path = os.path.join(
        args.output_path, f"{base_image_name}.txt")
    with open(merged_output_path, "w") as file:
        for detection in merged_detections:
            file.write(" ".join(map(str, detection[:5])) + "\n")

    return merged_output_path


def load_model(model_name, use_onnx, onnx_gpu):
    """
    Load the specified ONNX or PyTorch model for inference.

    Parameters:
    - model_name (str): Name of the model (excluding file extension).
    - use_onnx (bool): Flag indicating if the model is in ONNX format.
    - onnx_gpu (bool): Flag indicating if ONNX inference should use GPU.

    Returns:
    onnxruntime.InferenceSession or ultralytics.YOLO: Loaded model for inference.
    """

    # Load ONNX model
    if use_onnx:
        # Check if ONNX model file exists, if not, convert PyTorch model to ONNX
        if not os.path.exists(f"{model_name}.onnx"):
            model = YOLO(f"{model_name}.pt")
            model.export(format="onnx", opset=12, simplify=True, imgsz=640)

        # Load the model with appropriate execution provider (CPU or GPU)
        if onnx_gpu:
            session = ort.InferenceSession(f"{model_name}.onnx", providers=[
                                           'CUDAExecutionProvider', 'CPUExecutionProvider'])
        else:
            session = ort.InferenceSession(f"{model_name}.onnx", providers=[
                                           'CPUExecutionProvider'])

        return session

    # Load PyTorch model
    else:
        return YOLO(f"{model_name}.pt")
