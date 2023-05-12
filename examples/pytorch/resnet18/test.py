import numpy as np
import cv2
from rknn.api import RKNN
# import torchvision.models as models
import torch


# def export_pytorch_model():
#     net = models.resnet18(pretrained=True)
#     net.eval()
#     trace_model = torch.jit.trace(net, torch.Tensor(1,3,224,224))
#     trace_model.save('./resnet18.pt')


def show_outputs(output):
    output_sorted = sorted(output, reverse=True)
    top5_str = '\n-----TOP 5-----\n'
    for i in range(5):
        value = output_sorted[i]
        index = np.where(output == value)
        for j in range(len(index)):
            if (i + j) >= 5:
                break
            if value > 0:
                topi = '{}: {}\n'.format(index[j], value)
            else:
                topi = '-1: 0.0\n'
            top5_str += topi
    print(top5_str)


def show_perfs(perfs):
    perfs = 'perfs: {}\n'.format(perfs)
    print(perfs)


def softmax(x):
    return np.exp(x)/sum(np.exp(x))


if __name__ == '__main__':
    assert torch.__version__.startswith("1.10")

    # export_pytorch_model()

    # model = './resnet18.pt'
    # model = "/home/wenyu/fabric-ml/cloud/models/pytorch1.10/scriptmodule_cpu.pt"
    # model = "/home/wenyu/fabric-ml/cutpaste/models/model.pt"
    # model = "/home/wenyu/fabric-ml/cloud/outputs_edge/2023-05-11/02-49-31/traced.pt"
    model = "/home/wenyu/fabric-ml/cloud/outputs_edge/2023-05-12/03-36-00/traced.pt"
    input_size_list = [[3, 256, 256]]

    # Create RKNN object
    rknn = RKNN()

    # pre-process config
    print('--> Config model')
    rknn.config(reorder_channel='1 2 3', mean_values=[[0, 0, 0]],
                std_values=[[1, 1, 1]],)
    print('done')

    # Load Pytorch model
    print('--> Loading model')
    ret = rknn.load_pytorch(model=model, input_size_list=input_size_list)
    if ret != 0:
        print('Load Pytorch model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False) # to match up with pytorch, don't enable do_quantization
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn('./resnet_18.rknn')
    if ret != 0:
        print('Export resnet_18.rknn failed!')
        exit(ret)
    print('done')

    # ret = rknn.load_rknn('./cutpaste.rknn')

    # Set inputs
    img = cv2.imread('./space_shuttle_224.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    # edge_pair_input = np.random.randn((2, 3, 256, 256))
    # edge_piar_mask = np.zeros((1, 1281, 1281), dtype=bool)

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    # outputs = rknn.inference(inputs=[edge_pair_input, edge_piar_mask])
    print(outputs)
    rknn.release()

    # rknn and pytorch should match
    pytorch_model = torch.jit.load(model)
    pytorch_model.eval()
    input_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    print(pytorch_model(input_tensor))
