# Seaweed-Koneksi
Semantic segmentation models for seaweed cultivation based on remote sensing imagery

## Project Overview
This project focuses on the development of semantic segmentation models for identifying seaweed cultivation areas using remote sensing images. The benchmark dataset and models aim to provide a foundation for researchers and practitioners in the field of precision aquaculture and environmental monitoring.

## Benchmark Dataset
![PREVIEW BENCHMARK](https://github.com/user-attachments/assets/417bc8b8-b73a-476a-8312-e29a248197e8)

The benchmark dataset consists of remote sensing images from three different sources:

1. **Sentinel-2** : <br/> Multispectral imagery with a spatial resolution of 10 meters. The processing level is L2A or Bottom-of-atmosphere (BOA) reflectance. The channels include `B4, B3, B2, and B8` (Red, Green, Blue, and NIR).
2. **PlanetScope** : <br/> High-resolution imagery with a spatial resolution of 3 meters. The PlanetScope SuperDove `(PSB.SD)` sensor has been corrected to surface reflectance and harmonized. Here we use an 8-channel PlanetScope image, which we then reduce so that only `B2, B3, and B6` (Blue, Green, and NIR) channels remain.
3. **Pleiades** : <br/> Very high-resolution imagery with a spatial resolution of 2 meters. We have applied the orthorectification process to this image. Pleiades imagery includes four channels: `B1, B2, B3, and B4` (Red, Green, Blue, and NIR).

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
For model training, we use several types of encoders such as ResNet34 and Efficient-Net. Then, for model optimization we use the Adaptive Moment Estimation (ADAM) optimization algorithm and combine it with various loss functions such as Binary Cross Entropy (BCE) or Dice Loss. Model performance is calculated based on several evaluation metrics such as accuracy, loss, and Intersection over Union (IoU). 
We summarize the model training results from each benchmark dataset by showing the top three model performances below. 

| Benchmark   | Model          |Encoder         | Accuracy (Training) | Loss (Training) | IoU (Training) | Accuracy (Testing) | Loss (Testing) | IoU (Testing) |
|-------------|----------------|----------------|---------------------|-----------------|----------------|--------------------|----------------|---------------|
| Sentinel-2  | U-Net          | None           | 99.35%              | 0.02            | 84.51%         | 97.39%             | 0.16           | 44.03%        |
| Sentinel-2  | U-Net++        | ResNet34       | 98.82%              | 0.17            | 73.69%         | 97.21%             | 0.46           | 40.75%        |
| Sentinel-2  | DeepLabv3+     | ResNet34       | 98.65%              | 0.19            | 70.53%         | 97.01%             | 0.47           | 40.26%        |
| PlanetScope | U-Net          | None           | 98.42%              | 0.04            | 82.73%         | 94.11%             | 0.29           | 41.29%        |
| PlanetScope | U-Net++        | Effecient-Net  | 96.95%              | 0.20            | 68.87%         | 94.21%             | 0.40           | 43.82%        |
| PlanetScope | TransU-Net     | Effecient-Net  | 96.70%              | 0.21            | 67.69%         | 93.78%             | 0.41           | 43.09%        |
| Pleiades    | DeepLabv3+     | Effecient-Net  | 99.42%              | 0.05            | 92.17%         | 98.73%             | 0.11           | 83.04%        |
| Pleiades    | U-Net++        | Effecient-Net  | 99.25%              | 0.06            | 90.04%         | 98.67%             | 0.11           | 82.73%        |
| Pleiades    | PAN            | Effecient-Net  | 99.42%              | 0.05            | 92.19%         | 98.67%             | 0.11           | 82.53%        |


### Visualization
Segmentation results for each model are visualized to demonstrate their ability to identify seaweed cultivation areas. The visualizations include each images and model predictions.

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

