# Neural Network based deep text detector and recognizer
This module performs CRAFT based text detection and recognition using two pre-trained weights - craft and refine network. <a href='https://arxiv.org/abs/1904.01941'> Reference </a>

## Suitability

This module is suitable for text-detection on optical characters including screenshots (arising from screen captures, optically embedded text on images, etc), natural text in the wild, handwritten texts and digits. The capabilities of this module stem from domain-specific data-preprocessing which could be extended externally before using the detection module and also on hyper-parameter tuning internally in the network itself. 


## Usage of main module: find_text

Parameters Invoked | Default Argument | Description
--- | --- | ----
im | **Required** | Image variable (Type: np.ndarray)
threshold_txt | 0.7 | Fraction representing the text confidence threshold (Domain-specific)
threshold_link | 0.4 | Fraction representing the link confidence threshold (Domain-specific)
low_text | 0.4 | Lower bound score for the text (Domain-specific)
cuda | False | If enabled, the network will make use of GPU through CUDA interface
canvas_size | 1280 | Default image size used. (optional, the results won't vary unless the image is drastically larger or smaller)
zoom | 1.5 | Zoom-in factor (depends on how corrupted the image is, standard value is between 1.2-1.6)
poly | True | If set true, the boxes returned are both rectangle and polygonal
craft | True | If set false, craft network is not used for making inferences.
refine_net | True | If set false, refine network is not used for making inferences.
correction | False | Should pre-processing or image based corruption be applied on the data
boxes_only | False | If set true, the module makes only detection and not recognition. Ideal for scenarios where the module's recognition may not be a good fit, but detection is handy to be fit into the pipeline. 
              
              
```python
# Required packages (general)
from matplotlib.image import imread
from inferencer.detector import find_text

# Demo-specific packages (not required for the working of this module) 
from matplotlib import pyplot as plt
```


```python
# Can handle both image variable and image location
im = imread('dataset/live_me_screen.jpeg') # Sending input as an image array
plt.imshow(im)
```


![png](output_1_1.png)


```python
# Predictiion with both craft and refine net
print ("Detected Text = {}".format(find_text(im)))
```

    Detected Text =     Carrie  Magic Wand Oem umae i air  Bernard wanna join the prank  Carrie sends Magic Wand Say something 


```python
# Prediction with only craft
print ("Detected Text = {}".format(find_text(im, refine_net=False)))
```

    Detected Text = Broadcast video  live  Follow    Carrie  Magic eye lc Fanny Cannot  believe   air Bernard join  eyelale wanna Carrie Wand sends Magic Say something 
    
