# Seaweed-Koneksi
Semantic segmentation models for seaweed cultivation based on remote sensing imagery

## Project Overview
This project focuses on the development of semantic segmentation models for identifying seaweed cultivation areas using remote sensing images. The benchmark dataset and models aim to provide a foundation for researchers and practitioners in the field of precision aquaculture and environmental monitoring.

## Benchmark Dataset
The benchmark dataset consists of remote sensing images from three different sources:

1. **Sentinel-2** : Multispectral imagery with a spatial resolution of 10 meters.
2. **PlanetScope** : High-resolution imagery with a spatial resolution of 3 meters.
3. **Pleiades** : Very high-resolution imagery with a spatial resolution of 0.5 meters.

Each dataset is provided in `.tiff` format with an image size of `128x128` pixels. The dataset includes corresponding label files for each image, where:

- **Class 0**: Non-Seaweed
- **Class 1**: Seaweed

The labels were generated using the [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything), ensuring high-quality segmentation of seaweed cultivation areas.

## Model Training
To evaluate the benchmark dataset, we trained multiple deep learning architectures for semantic segmentation. The models include:

- **U-Net**
- **U-Net++**
- **DeepLabv3+**
- **PAN (Pyramid Attention Network)**
- **MANet (Multi-scale Attention Network)**
- **TransUNet (U-Net with Transformers)**

Some of these architectures are implemented using the [Segmentation Models PyTorch](https://github.com/qubvel-org/segmentation_models.pytorch) library. Each model was trained and evaluated on the benchmark dataset to determine its performance in segmenting seaweed cultivation from remote sensing images.

## Results
### Training metrics
The training results of each model are summarized below. The performance metrics include accuracy, loss, and Intersection over Union (IoU) for both seaweed and non-seaweed classes.


| Benchmark   | Model          | Accuracy (Training) | Loss (Training) | IoU (Training) | Accuracy (Testing) | Loss (Testing) | IoU (Testing) |
|-------------|----------------|---------------------|-----------------|----------------|--------------------|----------------|---------------|
| Sentinel-2  | U-Net          | 92.3%               | 0.15            | 85.7%          | 90.1%              | 0.18           | 83.2%         |
| Sentinel-2  | DeepLabv3+     | 93.5%               | 0.12            | 87.9%          | 91.4%              | 0.16           | 85.0%         |
| PlanetScope | U-Net++        | 91.0%               | 0.17            | 84.2%          | 88.5%              | 0.20           | 81.6%         |
| Pleiades    | TransUNet      | 94.2%               | 0.10            | 89.3%          | 92.7%              | 0.13           | 87.8%         |


- **Accuracy**
- **IoU (Intersection over Union)**
- **Dice Coefficient**

A detailed comparison of these metrics across datasets and models is provided in the results folder.


### Visualization
Segmentation results for each model are visualized to demonstrate their ability to identify seaweed cultivation areas. The visualizations include ground truth labels, model predictions, and overlay comparisons.

## Model Inference
Inference can be performed on new remote sensing images using the trained models. The output can be saved as:

1. **Geo-referenced raster files** (with coordinate transformations)
2. **PNG images**

## Getting Started
### Requirements
- Python 3.8+
- PyTorch
- torchvision
- segmentation-models-pytorch
- GDAL (for raster processing)

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/seaweed-segmentation.git
   cd seaweed-segmentation
   ```
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Running Inference
To run inference on a new image:
1. Prepare the input image in `.tiff` format.
2. Run the inference script:
   ```bash
   python inference.py --input your_image.tiff --model trained_model.pth --output output_file.png
   ```

## Contributions
Contributions to improve the dataset, models, or training pipeline are welcome. Feel free to submit issues or pull requests.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- Sentinel-2, PlanetScope, and Pleiades datasets for providing the imagery.
- Segment Anything Model for label generation.
- PyTorch Segmentation Models library for model implementations.

