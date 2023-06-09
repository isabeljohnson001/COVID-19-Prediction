# COVID-19-Prediction

### Getting Started

Download the covid-detection-from-chest xray.ipynb in Jupiter notebook.Download the Kaggle API token to the current directory.

### About the dataset

In this project, we will work with a dataset containing a posteroanterior (PA) view of chest X-ray images comprising Normal, Viral, and CVOID-19 affected patients. There are total 1823 CXR images.We propose COVIDLite, a deep neural network-based diagnostic method using CXR images, combining image preprocessing and DSCNN for accurate COVID-19 and viral pneumonia detection, outperforming state-of-the-art methods.

The dataset is taken from:-
[https://www.kaggle.com/datasets/arezalo/customer-dataset](https://www.kaggle.com/datasets/sid321axn/covid-cxr-image-dataset-research)

### Data Preparation
The data preparation process involves enhancing the chest X-ray images using white balance adjustment and Contrast Limited Adaptive Histogram Equalization (CLAHE) to improve image quality and contrast. The images are then resized to a fixed size of 224x224 pixels.

### Model Architecture
The model architecture is based on a series of convolutional and separable convolutional layers, followed by batch normalization, max pooling, and dropout layers. The model also includes fully connected layers for classification. The model is compiled with the Adam optimizer and the sparse categorical cross-entropy loss function.

### Training and Evaluation
The model is trained using the training dataset and evaluated using the evaluation dataset. During training, data augmentation techniques such as rotation are applied to improve model generalization. The training is performed for a fixed number of epochs with a batch size of 16. Model checkpoints and early stopping callbacks are used to save the best model weights.

#### Confusion Matrix: 
<kbd>
<img src=https://github.com/isabeljohnson001/COVID-19-Prediction/blob/main/confuison-matrix.png>
</kbd>

#### Classification Report: 
The classification report provides an evaluation of the performance of a classification model.The model performs well in general, with high precision, recall, and F1-scores for most classes, but it may require further improvement to enhance its ability to detect instances of the "virus" class accurately.

<kbd>
<img src=https://github.com/isabeljohnson001/COVID-19-Prediction/blob/main/class-report.png>
</kbd>

#### Model Prediction: 
A selection of correctly predicted and incorrectly predicted images from the evaluation dataset is displayed to provide a visual representation of the model's performance. Each image is shown along with its predicted label and the actual label for comparison.

<kbd>
<img src=https://github.com/isabeljohnson001/COVID-19-Prediction/blob/main/model-pred.png>
<img src=https://github.com/isabeljohnson001/COVID-19-Prediction/blob/main/model-pred-2.png>
</kbd>

