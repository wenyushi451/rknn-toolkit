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
    ret = rknn.load_onnx(model=ONNX_MODEL, inputs=["input"], input_size_list=[[3, 224, 224]], outputs=["cls", "rot_deg"])
    if ret != 0:
        print('Load cutpaste failed!')
        exit(ret)
    print('done')
    print('--> Building model')
    ret = rknn.build(do_quantization=False,pre_compile=False)
    if ret != 0:
        print('Build cutpaste failed!')
        exit(ret)
    print('done')
    ret = rknn.load_rknn(RKNN_MODEL,load_model_in_npu=False)
    if ret != 0:
        print("load rknn fail")
    print("done")

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL)
    print(ret)
    if ret != 0:
        print('Export cutpaste.rknn failed!')
        exit(ret)
    print('done')

    # init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime(device_id="0")
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')
