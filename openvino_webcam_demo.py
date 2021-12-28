import cv2
import numpy as np
import time
from skimage.filters import gaussian
import argparse

from openvino.runtime import Core
from openvino.preprocess import PrePostProcessor, ColorFormat, ResizeAlgorithm
from openvino.runtime import Type, Layout

def sharpen(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=5, multichannel=True)

    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)


def hair(image, parsing, part=17, color=[230, 50, 20]):
    b, g, r = color      #[10, 50, 250]       # [10, 250, 10]
    tar_color = np.zeros_like(image)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

    if part == 12 or part == 13:
        image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
    else:
        image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]

    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    if part == 17:
        changed = sharpen(changed)

    changed[parsing != part] = image[parsing != part]
    return changed

#infer one frame. preprocessing is inside model graph
def process_frame(frame):
    tic=time.time()
    frame=np.expand_dims(frame, 0) #HWC->#NHWC
    outputs=compiled_model.infer_new_request({0: frame})
    infer_time=(time.time() - tic)
    outputs=list(outputs.values())
    
    return infer_time, outputs[0].squeeze(0)

parser = argparse.ArgumentParser(description='OV_MU')
parser.add_argument('--device', default="CPU", help='Device to perform inference on')
args = parser.parse_args()

model_path='FaceParsing.xml'

#this is needed to know which image size will be passed to model
#we simply grab from camera and pass as is
camera=cv2.VideoCapture(0)
ret, frame=camera.read() 
input_h, input_w, _ = frame.shape
camera.release

#OpenVINO model wrangling
core=Core()

print("Devices available for OpenVINO inference:")
for device in core.available_devices:
    print(f'- {device}')

print("Demo will run on "+args.device)

model=core.read_model(model_path)

#preprocessing that will be integrated into compiled model
#executed for every inference call on target device
pp=PrePostProcessor(model)
#input tensor corresponds to what we have got from camera
pp.input().tensor() \
    .set_element_type(Type.u8) \
    .set_layout(Layout('NHWC')) \
    .set_spatial_static_shape(input_h, input_w) \
    .set_color_format(ColorFormat.BGR, [])
#convert from BGR to RGB
pp.input().preprocess().convert_color(ColorFormat.RGB)
#resize to model input size
pp.input().preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR)
#this will convert layout from NHWC to NCHW 
pp.input().model().set_layout(Layout('NCHW'))
#integrate preprocessing into model
model=pp.build()

#compile model on target device
compiled_model = core.compile_model(model, args.device)

table = {
    'hair': 17,
    'upper_lip': 12,
    'lower_lip': 13
}

camera=cv2.VideoCapture(0)
while True:
    ret, frame=camera.read() #BGR
    if not ret:
        print('Capture error')
        break

    infer_time, output=process_frame(frame) 
    parsing = output.argmax(0)

    parsing = cv2.resize(parsing, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    parts = [table['hair'], table['upper_lip'], table['lower_lip']]

    colors = [[230, 50, 20], [20, 70, 180], [20, 70, 180]]

    for part, color in zip(parts, colors):
        frame = hair(frame, parsing, part, color)

    infer_time = "OpenVINO inference time is "+ '{:.0f} ms'.format(infer_time*1000)
    cv2.putText(frame, infer_time, (0, 20),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    cv2.imshow('output', frame)

    key=cv2.waitKey(33)
    if (key%256==27) :
        break

camera.release
cv2.destroyAllWindows()
