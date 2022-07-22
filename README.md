# Breast Cancer Classification using Deep Learning

The purpose of this study is to investigate ways of fostering a deep learning model that can precisely diagnose breast cancer so that late treatments may be avoided due to false negatives and wasteful treatments can be avoided due to false positives. A dataset of 1312 images comprising both the types of images i.e. Benign and Malignant have been utilized. ResNet50, DenseNet201, AlexNet, and VGG16 models are implemented. The highest accuracy for our base model without any transfer learning is achieved. Augmentation is used to rescale the data and decrease the overfitting. The proposed method produces promising results when tested on a well-known publicly accessible dataset. Likewise, pre-trained models are effective in medical image analysis hence it performs precisely with a low loss function.

<p align="center"><img width="402" alt="image" src="https://user-images.githubusercontent.com/79443588/180518069-fbd21aae-448d-4d3c-845d-e9c63e9e34de.png"></p>

**Data Pre-processing and Augmentation**

The images are resized by 128 x 128 pixels to achieve better accuracy. For enhanced performance, all images are rescaled by dividing to RGB range 255 and separating labels. Likewise, mask images are considerable features in the model and my calculations yielded appropriate results according to the respective model. The normal images are easy to classify so it has not been considered further. There are total 3057 images after implementing data augmentation with required flow.

 

**Outline**

In the proposed baseline model to extract important features from the images, I have utilized three convolution layers with 16, 32, and 64 filters respectively with the input feature shape of 128 x 128 x 1 and MaxPooling2D for dimensionality reduction. I included a flatten layer to convert the extracted features into a 1D array. Later, Dense layers with 128, 64, and 32 units are added. To reduce overfitting, a dropout layer with a rate of 0.3 is followed up. For optimization of activation function, adaptive moment estimation is used. Likewise, the approach of feature engineering and hyperparameter tuning is performed to yield better parameters. Further, transfer learning models are implemented to validate the research.

**Overview on results**

<p align="center"><i><b>Averaged accuracy of all models with transformed original dataset</b></i></p>
<p align="center"><img width="750" alt="image" src="https://user-images.githubusercontent.com/79443588/180517353-74793826-d28c-4327-9b8a-930d182ddfcd.png"></p>

<p align="center"><i><b>Averaged accuracy of models with augmented data</b></i></p>
<p align="center"><img width="514" alt="image" src="https://user-images.githubusercontent.com/79443588/180517661-a84c8d6c-9436-4f21-ab89-6bc3efed7e48.png"></p>

**Conclusion**

The results of the classification showed that the proposed design has potential as far as different known quality metrics. In this approach, the implementation of transfer learning and a custom baseline model assist early and programmed detection of breast cancer from ultrasound images. In the dataset, there are 891 benign and 421 malignant cases. In the wake of implementation and assessment, the observation conveys that the best-performing models are Base Model, AlexNet, and ResNet50. Later, the model can be made more robust and precise by tuning the models and increasing the size of the dataset by utilizing new ultrasound images gathered from various locations and medical equipment.

The Breast Ultrasound Images Dataset was gathered by Dr. Aly Fahmy and his group. To download the original BUSI dataset. *[click here](https://scholar.cu.edu.eg/Dataset_BUSI.zip)*

For software requirements, the system should be exposed to [anaconda environment](https://www.anaconda.com/products/distribution) and required libraries should be installed.

Copyrights © 2022 by Neel Patel.
All rights are reserved.
