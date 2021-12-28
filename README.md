# RetinaFace in OpenVINO

This is OpenVINO based demo for face makeup. Forked from original repository here: [https://github.com/zllrunning/face-makeup.PyTorch](https://github.com/zllrunning/face-makeup.PyTorch)

Torchvision and PILdependency for demo has been eliminated and code can run without it. Some postprocessing steps are kept as is with minimal optimizations performed.

## Steps to reproduce

### Clone source and install packages
```Shell
https://github.com/yury-gorbachev/OpenVINO_FaceMakeup
conda create --name ov_makeup python=3.7
conda activate ov_makeup
cd OpenVINO_FaceMakeup
pip install -r requirements.txt
```

### Install OpenVINO and development tools
```Shell
pip install openvino-dev >= 22.1
```

### Convert model to onnx (Will generate MobileNet baset model by default)

```Shell
python ./onnx_export.py
```
This will generate FaceParsing.onnx

### Convert model to OpenVINO IR (Will generate MobileNet baset model by default)

```Shell
mo --use_new_frontend --mean_values="[123.675, 116.28, 103.53]" --scale_values="[58.395, 57.120000000000005, 57.375]" --input_model=FaceParsing.onnx
```
This will generate FaceParsing.xml and *.bin files that represent model in OV IR.
Mean values are integrated in model graph, no need for additional preprocessing.

### Run demo (requries camera)
```Shell
python openvino_webcam_demo.py
```
Demo will run on CPU by default with first available video capture device.

## Using different inference device

Use --device option to change device that will be used for inference (e.g. GPU). List of available devices is printed upon demo run.
Check out [openvino.ai](openvino.ai) for information on configuration

```Shell
python openvino_webcam_demo.py --device=GPU
```