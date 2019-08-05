# Udacity Self-Driving Car Engineer Nanodegree
## Project: Build a Traffic Sign Recognition Classifier Using Keras
### In stead of using Tensorflow as requried, I am implementating a classifer for recognizing traffic sign using deep learning and convolution neural network with Keras. 

## Getting Started
In order to run the script, Anaconda is recommended to be installed for running Jupyter Notebook. A seperated virtual environment is also recommended to be created for isolation with other projects. You can find more information from this [Doc](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

### Prerequisites

There are several packages needed to be installed. ```pip install PACKAGE ``` command is handy to use. For example, 

```
pip install tensorflow
pip install tensorflow-gpu (if you have a GPU)
pip install keras
pip install sklearn
pip install matplotlib
pip install csv
pip install pickle
pip install numpy
pip install opencv-python
```

## Working Pipeline
### 1. Data Loading
- We have to load image data from directory.<br> In the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset), we need [training](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip) and [testing](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip) dataset in this project. Since there is no validation set provided, we are going to split training data into 2 parts for training and validation.<br> 
- Please specify the **TRAING_DIR** and **TESTING_DIR** path from your computer.<br>
- In Windows, for example, TRAING_DIR = r'.\UdacityTrafficSign\GTSRB_Final_Training_Images\GTSRB\Final_Training\Images'
    
### 2. Preprocessing
- Preprocessing is a stage to regularize the input data in order to reduce the computation complexity and increase the training speed. In this project, we will apply 2 preprocessing techniques:
    - **Gray Scale Transformation**, transforming RGB channels to gray channel, and
    - **Normalization**, translating the whole dataset to a 0-mean distribution by **f = (data - mean)/std** .
    
### 3. Network Architecture
- In my network, there are 3 levels of convolution layers with activation and max_pooling process to extract the features. Fully connected layer and dropout layer are applied for generalization. Finally, we can output a softmax dense layer to classify the input object with 43 (number of classes) probabilities.

|Layer |  Output Shape|
|:------| :------------|
|Input | (None, 32, 32, 1)|
|Convolution (32, 3x3, valid, ReLU)| (None, 30, 30, 32)|
|BatchNormalization| (None, 30, 30, 32)|
|Max Pooling (2x2,valid)| (None, 15, 15, 32)|
|Convolution (32, 3x3, valid, ReLU)| (None, 13, 13, 32)|
|BatchNormalization| (None, 13, 13, 32)|
|Max Pooling (2x2,valid)| (None, 6, 6, 32)|
|Convolution (32, 3x3, valid, ReLU)| (None, 4, 4, 32)|
|BatchNormalization| (None, 4, 4, 32)|
|Max Pooling (2x2,valid)| (None, 2, 2, 32)|
|Flatten| (None, 128)|
|Dense (ReLU)| (None, 64)|
|Dropout (0.5)| (None, 64)|
|Dense (Softmax)| (None, 43)|

### 4. Model
- Optimizer: Adam 
- Learning rate: 0.001
- Loss function: categorical_crossentropy
- Batch size: 500
- epoch: 100

### 4. Evaluation
- The network trained well with:
    - Training loss = 0.014
    - Training accuracy = 0.995
    - Validation loss = 0.028
    - Validation accuracy = 0.993
    - Overall accuracy = 0.99
- For testing, it obtained accuracy with 0.97 which is a good performance

### 5. Discussion
- Overftting and Generalization
    - The log of acc and loss of training and validation set shows that they are processing with a similiar way. It means that my network does not overfit too much. For the testing statge, my testing score is 0.96 which is 0.03 lower than training result. Notwithstanding it is not a perfest result, the performance of my the nerwork is acceptable for this traffic sign classifier dataset.
- Comparision of another network
    - I have compared my network with a solution proposed in GitHub with similiar training, validation and testing performance.The number of conv layers I used (4 vs 11), however, is around 3 times fewer than it. Faster learning procecess can be acheived with fewer numbers of parameters for my model.
- Effect of Batchnormalization
    - On top of the structure of MyNet_v2, I added batch normalization layer at each of the activation layers. It results an enhancement of 0.01 score. It is because it can normalize the input to the activation function so that the inpt data is centered to the linear section of the activation fucntion. In my case, the centre of ReLU function is 0. It generalizes the input so that it is easier to seperate the features. [(For More Information)](https://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras)
    
| |MyNet_v1|MyNet_v2|MyNet_v3|Another Network|
|:------|:------:|:------:|:------:|:------------:|
|# Conv Layer|3|4|4|11|
|Total params|169,131|47,307|47,819|412,715|
|Trainable params|169,131|47,307|47,563|412,715|
|Non-trainable params|0|0|256|0|
|Batchnomalization|No|No|Yes|No|
|Testing accuracy|0.95|0.96|0.97|0.96|



