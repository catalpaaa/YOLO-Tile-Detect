import argparse

parser = argparse.ArgumentParser(
    description="Run YOLO inference on image tiles")
parser.add_argument("--output_path", "-o", type=str, default="output",
                    help="Output directory for inference results")
parser.add_argument("--input_path", "-i", type=str, required=True,
                    help="Path to the input images")
parser.add_argument("--use_onnx", action="store_true",
                    help="Use ONNX model for inference instead of YOLO model")
parser.add_argument("--onnx_gpu", action="store_true",
                    help="Use GPU in addition to CPU for inferencing the ONNX model")
parser.add_argument("--model_name", type=str, required=True,
                    help="Name of the YOLO model")
parser.add_argument("--tile_overlap", type=float, default=0.25,
                    help="The percentage of the overlap around each tile")
parser.add_argument("--class_yaml", type=str, required=True,
                    help="Path to the YAML file with class names")
parser.add_argument("--max_tile_size", type=int, default=512,
                    help="Maximum pixel size of the image tiles")
parser.add_argument("--iou_threshold", type=float, default=0.07,
                    help="Intersection over Union threshold for merging bounding boxes")
parser.add_argument("--dist_threshold", type=float, default=0.03,
                    help="Distance threshold for merging bounding boxes")
parser.add_argument("--keep_temp", action="store_true",
                    help="Keep temporary folder after processing")
args = parser.parse_args()
