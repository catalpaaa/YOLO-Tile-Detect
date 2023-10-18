# Object Detection Across Image Tiles

## Introduction

This application uses YoloV8 to detect objects within images across tiles. Total runtime is printed for comparison between different model architectures.

## Optimization

- Model can be converted to ONNX format for faster inference speed on CPU/GPU.
- Implemented parallel computing to optimize postprocessing on multiple tiles.
- Introduced a buffer around each image to enhance detection accuracy.
- Set up a Docker environment to provide the app as a standalone executable.

## Usage

### Basic Usage

Install requirements

```bash
pip install -r requirements.txt
```

To run with pytorch. Default GPU, fallback to CPU.

```bash
python main.py --input_path "your/input/path/" --output_path "your/output/path/"
```

To run with ONNX using CPU only

```bash
python main.py --input_path "your/input/path/" --output_path "your/output/path/ --use_onnx"
```

To run with ONNX using GPU, fallback to CPU

```bash
python main.py --input_path "your/input/path/" --output_path "your/output/path/ --use_onnx --onnx_gpu"
```

### Sample command

Use PyTorch, runs on GPU, fallback to CPU

```bash
python main.py -i test/ --model_name yolov8x --class_yaml coco8.yaml
```

Use onnx, runs on CPU

```bash
python main.py -i test/ --model_name yolov8x --class_yaml coco8.yaml --use_onnx
```

Use onnx, runs on GPU, fallback to CPU

```bash
python main.py -i test/ --model_name yolov8x --class_yaml coco8.yaml --use_onnx --onnx_gpu
```
