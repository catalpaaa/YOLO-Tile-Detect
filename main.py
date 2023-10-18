import os
import glob
import shutil
import time

import cv2
from PIL import Image

from modules.args import args
from modules.utils import load_model, combine_tile_bboxes
from modules.onnx_utils import preprocess, postprocess
from modules.visualization import render_final_detections


def main():
    """
    Main function to perform object detection on input images using specified models.
    """

    # Load the desired model for inference
    model = load_model(args.model_name, args.use_onnx, args.onnx_gpu)

    # Begin timer to calculate total execution time
    start_time = time.time()

    # Retrieve all image paths from the specified directory
    image_paths = glob.glob(os.path.join(args.input_path, "*.png"))

    for image_path in image_paths:
        # Read the image and retrieve its dimensions
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape

        base_image_name = os.path.splitext(os.path.basename(image_path))[0]
        temp_output_folder = os.path.join("temp", base_image_name)

        # Ensure the temporary output directory exists
        if not os.path.exists(temp_output_folder):
            os.makedirs(temp_output_folder)

        # If using ONNX model, set the tile dimensions
        if args.use_onnx:
            tile_width = 640
            tile_height = 640

            # Get input shape information for ONNX model
            model_inputs = model.get_inputs()
            input_shape = model_inputs[0].shape
            input_width = input_shape[2]
            input_height = input_shape[3]
        else:
            # Calculate tile dimensions based on the image dimensions and the specified maximum tile size
            num_tiles_x = image_width // args.max_tile_size + \
                (1 if image_width % args.max_tile_size else 0)
            num_tiles_y = image_height // args.max_tile_size + \
                (1 if image_height % args.max_tile_size else 0)

            tile_width = image_width // num_tiles_x
            tile_height = image_height // num_tiles_y

        # Calculate the overlap between tiles for seamless stitching
        overlap_width = int(tile_width * args.tile_overlap)
        overlap_height = int(tile_height * args.tile_overlap)

        # Process each tile of the image and save the detection results
        for y in range(0, image_height - overlap_height, tile_height - overlap_height):
            for x in range(0, image_width - overlap_width, tile_width - overlap_width):
                # If using ONNX model, preprocess the tile and perform inference
                if args.use_onnx:
                    img_data = preprocess(
                        image[y: y + tile_height, x: x + tile_width], input_width, input_height)
                    outputs = model.run(None, {model_inputs[0].name: img_data})

                    # Post-process the ONNX model's outputs and save the result
                    tile_text_name = os.path.join(
                        temp_output_folder, f"{base_image_name}_y{y}_x{x}.txt")
                    output_img = postprocess(
                        image[y: y + tile_height, x: x + tile_width], outputs, input_width, input_height, 0.5, 0.5, tile_text_name)
                    tile_image_name = os.path.join(
                        temp_output_folder, f"{base_image_name}_y{y}_x{x}.png")
                    cv2.imwrite(tile_image_name, output_img)
                else:
                    # For non-ONNX models, predict directly on the tile
                    tile_prediction = model.predict(
                        image[y: y + tile_height, x: x + tile_width], conf=0.5)
                    tile_image_name = os.path.join(
                        temp_output_folder, f"{base_image_name}_y{y}_x{x}.png")
                    tile_text_name = os.path.join(
                        temp_output_folder, f"{base_image_name}_y{y}_x{x}.txt")
                    Image.fromarray(tile_prediction[0].plot(
                    )[..., ::-1]).save(tile_image_name)
                    tile_prediction[0].save_txt(tile_text_name)

        # Ensure the main output directory exists
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)

        # Combine bounding boxes from individual tiles
        merged_text_path = combine_tile_bboxes(
            temp_output_folder, base_image_name, image_width, image_height, tile_width, tile_height)

        # Render the final detections on the original image
        render_final_detections(merged_text_path, image_path, os.path.join(
            args.output_path, f"{base_image_name}.png"), args.class_yaml)

    # Calculate and print the total runtime
    end_time = time.time()
    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")

    # Clean up by removing temporary files if specified
    if not args.keep_temp:
        shutil.rmtree("temp")


if __name__ == "__main__":
    main()
