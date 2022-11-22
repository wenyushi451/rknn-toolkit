import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN
ONNX_MODEL = './cutpaste.onnx'
RKNN_MODEL = './cutpaste.rknn'


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()
    ONNX_MODEL="cutpaste.onnx"
    print('--> Config model')
    # rknn.config(mean_values=[[123.675, 116.28, 103.53]], std_values=[[58.82, 58.82, 58.82]], reorder_channel='0 1 2')
    print('done')

    print('--> Loading model')
    ret = rknn.load_rknn(path=RKNN_MODEL)
    if ret != 0:
        print('Load cutpaste failed!')
        exit(ret)
    print('done')
    
    # Set inputs
    img = cv2.imread('./dog_224x224.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    print(outputs)
