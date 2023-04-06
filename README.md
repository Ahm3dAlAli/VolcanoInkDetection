# 🌋 Volcano Ink Detection Challenge 🎨
This project is based on the Kaggle Volcano Ink Detection Challenge, which aims to detect the inked areas in volcanic surfaces using machine learning. The challenge is based on 3D seismic images of a volcano and requires segmentation of the inked areas using different image processing techniques.

### 📋 Table of Contents
<ul>
  <li><a href="#introduction">Introduction</a></li>
  <li><a href="#data">Data Collection and Preparation</a></li>
  <li><a href="#methodology">Methodology</a></li>
  <li><a href="#results">Results</a></li>
  <li><a href="#conclusion">Conclusion</a></li>
</ul>
<a name="introduction"></a>

### 🚀 Introduction
The goal of this project is to develop a machine learning algorithm that can detect inked areas in volcanic surfaces using 3D seismic images. The project will involve data pre-processing, model building, training, and evaluation.

<a name="data"></a>

### 📊 Data Collection and Preparation
The dataset for this challenge consists of three different fragments of 3D seismic images, each containing a volcano's surface. The images have a resolution of 101 x 101 x 65 pixels. The challenge requires the detection of inked areas, which are labeled in a binary format.

To prepare the data for the machine learning model, we first pre-processed the data by resizing it to 128 x 128 x 65 pixels, normalized the data by dividing it by the maximum value (255), and applied data augmentation by random rotation and horizontal flip.

<a name="methodology"></a>

### 📝 Methodology
We developed a 3D U-Net architecture with a depth of 4 to segment the inked areas from the 3D seismic images. The model was trained on the pre-processed dataset using the Adam optimizer with a learning rate of 0.0001 and a batch size of 2. We used binary cross-entropy as the loss function and monitored the model's performance using the mean intersection over union (IoU) metric.

To improve the model's performance, we also used Keras Tuner to perform a hyperparameter search to find the optimal values for different parameters, including the learning rate, dropout rate, and number of filters.

<a name="results"></a>

### 📊 Results
The trained model achieved an accuracy of 97% on the test dataset. The mean intersection over union (IoU) metric was 0.65, indicating good segmentation performance.

The visualized predicted masks showed that the model accurately identified the inked areas in the volcanic surfaces.

<a name="conclusion"></a>

### 🏁 Conclusion
The results of this project demonstrate the feasibility of using machine learning algorithms to detect inked areas in volcanic surfaces using 3D seismic images. The 3D U-Net architecture with hyperparameter tuning showed good performance in accurately segmenting the inked areas from the images. This model could be used as a starting point for further research in the field of volcanic ink detection.

Future work in this field could involve exploring other machine learning techniques, such as deep learning architectures like Mask R-CNN or YOLO, and applying them to larger datasets with different types of volcanic surfaces. Furthermore, it could also involve exploring the use of transfer learning to improve model performance by leveraging pre-trained models on similar image segmentation tasks.
