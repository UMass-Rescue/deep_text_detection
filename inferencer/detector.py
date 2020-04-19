# -*- coding: utf-8 -*-
from inferencer.utilities import fetch_models, resize_aspect_ratio, normalizeMeanVariance, getDetBoxes, adjustResultCoordinates
from matplotlib.image import imread
import cv2
import torch
from torch.autograd import Variable
import numpy as np

import sys
sys.path.append('../single_shot_text_detection/')
from inference import fetch_text_from_image

def automatic_brightness_and_contrast(image, clip_hist_percent=25):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result

def resolution_correction(im):
    pxmin = np.min(im)
    pxmax = np.max(im)
    imgContrast = (im - pxmin) / (pxmax - pxmin) * 255
    kernel = np.ones((2, 2), np.uint8)
    imgMorph = cv2.erode(imgContrast, kernel, iterations = 2)
    imgMorph = imgMorph.astype(np.uint8)
    imgMorph = cv2.resize(imgMorph, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return imgMorph

def evaluate_im(im, model_craft, model_refine_net=None, threshold_txt: float = 0.7, threshold_link: float = 0.4, 
                low_text: float = 0.4, cuda: bool = False, canvas_size: int = 1280, zoom: float = 2.0, poly: bool = True):
    
    resized_im, target_ratio, _ = resize_aspect_ratio(im, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=zoom)
    ratio_h = ratio_w = 1 / target_ratio
    x = normalizeMeanVariance(resized_im)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    with torch.no_grad():
        y, feature = model_craft(x)

    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    if model_refine_net is not None:
        with torch.no_grad():
            y_refiner = model_refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    boxes, _ = getDetBoxes(score_text, score_link, threshold_txt, threshold_link, low_text, poly)
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    return boxes

def compute_inference(im, correction=False):
    if correction:
        
    translation = fetch_text_from_image(im)[0]
    return translation

def find_text(im, threshold_txt: float = 0.7, threshold_link: float = 0.4, low_text: float = 0.4, 
              cuda: bool = False, canvas_size: int = 1280, zoom: float = 1.5, poly: bool = True,
              craft:bool = True, refine_net:bool = True, correction: bool = False, boxes_only: bool = False):
    
    if type(im) != np.ndarray:
        im = imread(im)
        
    model_craft, model_refine_net = fetch_models(cuda, craft, refine_net)
    compute_boxes = evaluate_im(im, model_craft, model_refine_net, threshold_txt, threshold_link, 
                            low_text, cuda, canvas_size, zoom, poly)
    
    predicted_text = []
    if boxes_only:
        return compute_boxes[1:]
    else:
        for box in compute_boxes[1:]:
            v_min, v_max = np.floor(np.min(box[:, 1])), np.floor(np.max(box[:, 1]))
            h_min, h_max = np.floor(np.min(box[:, 0])), np.floor(np.max(box[:, 0]))
            v_min, v_max, h_min, h_max = int(v_min-1), int(v_max-1), int(h_min-1), int(h_max-1)
            strip = im[v_min:v_max, h_min:h_max]

            # Make Inference
            text = compute_inference(strip, correction)
            predicted_text.append(text)

        return " ".join(predicted_text)