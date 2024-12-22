# Seaweed-Koneksi
Semantic segmentation models for seaweed cultivation based on remote sensing imagery

## Project Overview
This project is the part of KONEKSI project which is "Developing applications of satellite imagery for modelling environmental and social impacts of climate change on seaweed farming in Indonesia".
This research focuses on the development of semantic segmentation models for identifying seaweed cultivation areas using remote sensing images. We has compiled a multiscale benchmark dataset from various remote sensing platforms for mapping seaweed cultivation. This benchmark dataset serves as a foundational resource for future monitoring for sustainable seaweed cultivation mapping using deep learning techniques. However, collecting benchmark data poses significant challenges, as each type of imagery has its own strengths and limitations in terms of spatial, spectral, and temporal resolution. Consequently, creating high-quality seaweed labels requires specific treatments tailored to these characteristics. 

The results of this research are still being compiled into an article. This will be updated in the future.

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

## Data Access
The dataset can be downloaded below:
- [Sentinel-2](https://drive.google.com/file/d/19V_Vq6IHc2o5fz7prUpGYKsSFyBOxnJf/view?usp=sharing)
- [PlanetScope](https://drive.google.com/file/d/1XuiVnLhOY4yxAwRc7Ywb3c7btcXZ2JGw/view?usp=sharing)
- [Pleiades](https://drive.google.com/file/d/1VgnWq51-m0_pc7ZyDHpLhrTXwWa-24qS/view?usp=sharing)

However, we cannot share all of the datasets freely. This is related to the license of the image. Here the sentinel-2 imagery is freely accessible. But for PlanetScope and Pleiades images, please request access in advance.

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
Segmentation results for each model are visualized to demonstrate their ability to identify seaweed cultivation areas. The visualizations include each images and model predictions. Our visualization shows the results of the inference model using one of the testing data.
![model_inference_full_2](https://github.com/user-attachments/assets/64a51ba5-6af9-45c4-9a8b-80be43480c71)

## License
- This dataset is made available under the MIT license, freely available for both academic and commercial use. <br/>
- Access to Sentinel data is free, full and open for the broad Regional, National, European and International user community. View [Terms and Conditions](https://scihub.copernicus.eu/twiki/do/view/SciHubWebPortal/TermsConditions).
- Access to Planet data is restricted. View [Terms of Use](https://www.planet.com/terms-of-use/).
- The EULA for Pleiades imagery includes restrictions on how the imagery can be used. View [Terms and Conditions](https://space-solutions.airbus.com/legal/terms-and-conditions/)

## Contact
[email](mailto:koneksiseaweed@gmail.com)

## Acknowledgments
This research has been funded by the Department of Foreign Affairs and Trade Australia through KONEKSI. The views expressed in this research are the authors’ alone and are not necessarily the views of the Australian Government.

