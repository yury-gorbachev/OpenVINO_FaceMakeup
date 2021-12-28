import torch
from model import BiSeNet

model='cp/79999_iter.pth'

n_classes = 19
net = BiSeNet(n_classes=n_classes)

net.load_state_dict(torch.load(model, map_location=torch.device('cpu')))
net.eval()

input=torch.rand((1,3,512,512))
inputs = ['image']
outputs = ['processed_image']
torch.onnx.export(net, input, 'FaceParsing.onnx', input_names=inputs, output_names=outputs, opset_version=11 )