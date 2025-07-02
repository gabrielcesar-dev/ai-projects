# YOLO People Counter

A Python application that uses YOLO (You Only Look Once) deep learning model to detect and count people in images. This project provides a clean, organized structure for batch processing images and generating annotated results.

## Features

- 🔍 **People Detection**: Uses YOLO11 models to detect people in images
- 📊 **Batch Processing**: Process multiple images at once
- 🎯 **High Accuracy**: Configurable confidence and IoU thresholds
- 📁 **Organized Structure**: Clean separation of samples, results, and models
- 🖥️ **GPU Support**: CUDA acceleration for faster processing
- 📋 **Command Line Interface**: Easy-to-use CLI with multiple options
- 🔄 **Error Handling**: Robust error handling and informative output

## Project Structure

```
yolo-count/
├── main.py              # Main application script
├── samples/             # Input images directory
│   └── .gitkeep        # Keeps directory in git
├── results/             # Output annotated images
│   └── .gitkeep        # Keeps directory in git
├── models/              # YOLO model files
│   └── .gitkeep        # Keeps directory in git
├── pyproject.toml       # Project dependencies
├── uv.lock             # Lock file for dependencies
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for acceleration)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd yolo-count
   ```

2. **Install dependencies using uv** (recommended):
   ```bash
   uv sync
   ```

   Or using pip:
   ```bash
   pip install ultralytics opencv-python
   ```

3. **Download YOLO models**:
   ```bash
   # The models will be automatically downloaded on first use
   # Or manually download and place in models/ directory
   ```

## Usage

### Basic Usage

1. **Place your images** in the `samples/` directory
2. **Run the script**:
   ```bash
   python main.py
   ```

### Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --model PATH        Path to YOLO model (default: models/yolo11s.pt)
  --samples DIR       Directory containing sample images (default: samples)
  --results DIR       Directory to save results (default: results)
  --device DEVICE     Device for inference (default: cuda:0)
  --conf FLOAT        Confidence threshold (default: 0.3)
  --iou FLOAT         IoU threshold (default: 0.3)
  --show             Display processed images
  --single FILE      Process single image file
  --help             Show help message
```

### Examples

**Process all images in samples directory**:
```bash
python main.py
```

**Process with custom settings**:
```bash
python main.py --conf 0.5 --iou 0.4 --show
```

**Process a single image**:
```bash
python main.py --single path/to/image.jpg --show
```

**Use CPU instead of GPU**:
```bash
python main.py --device cpu
```

**Use different model**:
```bash
python main.py --model models/yolo11n.pt
```

## Model Information

This project supports YOLO11 models:

- **yolo11n.pt**: Nano model (fastest, least accurate)
- **yolo11s.pt**: Small model (balanced speed/accuracy) - Default
- **yolo11m.pt**: Medium model (more accurate, slower)
- **yolo11l.pt**: Large model (high accuracy, slow)
- **yolo11x.pt**: Extra Large model (highest accuracy, slowest)

Models are automatically downloaded from Ultralytics on first use.

## Output

The application generates:

1. **Annotated Images**: Saved in `results/` directory with bounding boxes and confidence scores
2. **Console Output**: Detection counts and processing summary
3. **Batch Summary**: Total people count across all processed images

### Sample Output

```
📁 Processing 3 images from samples
------------------------------------------------------------
✓ Processed: image1.jpg -> 2 people detected
✓ Processed: image2.jpg -> 5 people detected  
✓ Processed: image3.jpg -> 1 people detected
------------------------------------------------------------
📊 Summary: 8 total people detected across 3 images
```

## Configuration

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

### Performance Tuning

- **Confidence Threshold**: Lower values detect more objects but may include false positives
- **IoU Threshold**: Controls overlap tolerance in Non-Maximum Suppression
- **Device Selection**: Use `cuda:0` for GPU acceleration, `cpu` for CPU-only
- **Model Selection**: Balance between speed and accuracy based on your needs

## Development

### Adding New Features

The code is organized in a class-based structure for easy extension:

- `PeopleCounter` class: Main detection logic
- `process_image()`: Single image processing
- `process_batch()`: Batch processing logic
- `main()`: CLI interface

### Error Handling

The application includes comprehensive error handling for:

- Missing model files
- Invalid image files
- CUDA availability issues
- File I/O errors

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Use `--device cpu` or smaller batch sizes
2. **Model not found**: Ensure model files are in the `models/` directory
3. **No images detected**: Check image file extensions and directory paths
4. **Permission errors**: Ensure write permissions for `results/` directory

### Dependencies Issues

If you encounter dependency issues:

```bash
# Update ultralytics
pip install --upgrade ultralytics

# Reinstall OpenCV
pip install --force-reinstall opencv-python
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the YOLO implementation
- [OpenCV](https://opencv.org/) for image processing capabilities