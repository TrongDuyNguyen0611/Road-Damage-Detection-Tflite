# Convert Darknet YOLOv4 or YOLOv3 to TensorFlow Model
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

YOLOv4, YOLOv4-tiny Implemented in Tensorflow 2.3. 

This repository is created for the channel [TheCodingBug](https://www.youtube.com/channel/UCcNgapXcZkyW10FIOohZ1uA?sub_confirmation=1).

This repository shows how to convert YOLO v4, YOLOv3, YOLO tiny .weights to .pb, .tflite and trt format for tensorflow, tensorflow lite, tensorRT.


1. [Setting Up Environment](#setting-up-environment)
   * [Using Conda](#using-conda)
   * [Using Pip](#using-pip)
2. [Convert YOLOv4 to TensorFlow](#convert-yolov4-to-tensorflow)
   * [Run Object Detection](#run-object-detection)
3. [Convert YOLOv4 to tflite](#convert-to-tflite)
   * [Run Objection Detection using TFLite Model](#run-object_detection-using-tflite-model)
4. [FPS Comparison](#fps-comparison)



# Setting Up Environment
### Using Conda
#### CPU
```bash
# CPU
conda env create -f conda-cpu.yml


# activate environment on Windows or Linux
conda activate tf-cpu

# activate environment on Mac
source activate tf-cpu
```
#### GPU
```bash
# GPU
conda env create -f conda-gpu.yml

# activate environment on Windows or Linux
conda activate tf-gpu

# activate environment on Mac
source activate tf-gpu
```

### Using Pip
```bash
# CPU
pip install -r requirements.txt

# GPU
pip install -r requirements-gpu.txt

```
**Note:** If installing GPU version with Pip, you need to install CUDA and cuDNN in your system. You can find the tutorial for Windows [here](https://www.youtube.com/watch?v=PlW9zAg4cx8).
# Performance
<p align="center"><img src="data/performance.png" width="640"\></p>

# Download Weights File
Download `yolov4.weights` file 245 MB: [yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) (Google-drive mirror [yolov4.weights](https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT) )

If using ``tiny`` version, download [yolov4-tiny.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights) file instead. I am using original ``yolov4`` because its more accurate (although slower).

# Convert YOLOv4 to TensorFlow

```bash
# Convert darknet weights to tensorflow
## yolov4
python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4 

## yolov4-tiny
python save_model.py --weights ./data/yolov4-tiny.weights --output ./checkpoints/yolov4-tiny-416 --input_size 416 --model yolov4 --tiny

```
If you want to run yolov3 or yolov3-tiny change ``--model yolov3`` in command and also download corresponding YOLOv3 weights and and change ``--weights to ./data/yolov3.weights``

### Run Object Detection

```bash
# Run yolov4 on image
python detect.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --image ./data/kite.jpg

python detect.py --weights ./checkpoints/yolov4-tiny-416 --size 416 --model yolov4 --image ./data/kite.jpg --tiny

# Run on multiple images
python detect.py --weights ./checkpoints/yolov4-tiny-416 --size 416 --model yolov4 --images "./data/kite.jpg, ./data/girl.jpg"

# Run yolov4 on video
python detect_video.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --video ./data/japan.mp4 --output ./detections/video_output.avi

# Run yolov4 on webcam
python detect_video.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --video 0 --output ./detections/webcam_output.avi

```

#### Output

##### Yolov4 original weight
<p align="center"><img src="result.png" width="640"\></p>

# Convert to tflite

```bash
# Save tf model for tflite converting
python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4 --framework tflite

# yolov4
python convert_tflite.py --weights ./checkpoints/yolov4-416 --output ./checkpoints/yolov4-416.tflite

# yolov4 quantize float16
python convert_tflite.py --weights ./checkpoints/yolov4-416 --output ./checkpoints/yolov4-416-fp16.tflite --quantize_mode float16

# yolov4 quantize int8
python convert_tflite.py --weights ./checkpoints/yolov4-416 --output ./checkpoints/yolov4-416-int8.tflite --quantize_mode int8

```
### Run Objection Detection using TFLite Model

```bash
# Run demo tflite model
python detect.py --weights ./checkpoints/yolov4-416.tflite --size 416 --model yolov4 --image ./data/kite.jpg --framework tflite

# Run demo tflite on videos
python detect_video.py --weights ./checkpoints/yolov4-416.tflite --size 416 --model yolov4 --video ./data/japan.mp4 --output ./detections/video_output.avi --framework tflite

```
##### Yolov4 tflite int8 weight
<p align="center"><img src="result-int8.png" width="640"\></p>

Yolov4 and Yolov4-tiny int8 quantization have some issues. I will try to fix that. You can try Yolov3 and Yolov3-tiny int8 quantization 
# Convert to TensorRT
Similar to above method, you can convert darknet ``yolov3`` or ``yolov4`` models to ``tensorflow`` and then to ``TensorRT``.

```bash
# yolov3
python save_model.py --weights ./data/yolov3.weights --output ./checkpoints/yolov3.tf --input_size 416 --model yolov3
python convert_trt.py --weights ./checkpoints/yolov3.tf --quantize_mode float16 --output ./checkpoints/yolov3-trt-fp16-416

# yolov3-tiny
python save_model.py --weights ./data/yolov3-tiny.weights --output ./checkpoints/yolov3-tiny.tf --input_size 416 --tiny
python convert_trt.py --weights ./checkpoints/yolov3-tiny.tf --quantize_mode float16 --output ./checkpoints/yolov3-tiny-trt-fp16-416

# yolov4
python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4.tf --input_size 416 --model yolov4
python convert_trt.py --weights ./checkpoints/yolov4.tf --quantize_mode float16 --output ./checkpoints/yolov4-trt-fp16-416
```
# FPS Comparison

We use ``japan.mp4`` for all the experiments and report following results.

**OpenCV GPU:** 9.96 FPS

**Darknet GPU:** 13.5 FPS

**TensorFlow GPU:** 11.5 FPS

**TFLite CPU:** 2 FPS

**TFLite GPU (On Android):** To be tested

# Evaluate on COCO 2017 Dataset
```bash
# run script in /script/get_coco_dataset_2017.sh to download COCO 2017 Dataset
# preprocess coco dataset
cd data
mkdir dataset
cd ..
cd scripts
python coco_convert.py --input ./coco/annotations/instances_val2017.json --output val2017.pkl
python coco_annotation.py --coco_path ./coco 
cd ..

# evaluate yolov4 model
python evaluate.py --weights ./data/yolov4.weights
cd mAP/extra
python remove_space.py
cd ..
python main.py --output results_yolov4_tf
```
#### mAP50 on COCO 2017 Dataset

| Detection   | 512x512 | 416x416 | 320x320 |
|-------------|---------|---------|---------|
| YoloV3      | 55.43   | 52.32   |         |
| YoloV4      | 61.96   | 57.33   |         |


# Traning your own model
```bash
# Prepare your dataset
# If you want to train from scratch:
In config.py set FISRT_STAGE_EPOCHS=0 
# Run script:
python train.py

# Transfer learning: 
python train.py --weights ./data/yolov4.weights
```
The training performance is not fully reproduced yet, so I recommended to use Alex's [Darknet](https://github.com/AlexeyAB/darknet) to train your own data, then convert the .weights to tensorflow or tflite.



# TODO
* [x] Convert YOLOv4 to TensorRT
* [x] YOLOv4 tflite on android
* [ ] YOLOv4 tflite on ios
* [x] Training code
* [x] Update scale xy
* [ ] ciou
* [ ] Mosaic data augmentation
* [x] Mish activation
* [x] yolov4 tflite version
* [x] yolov4 in8 tflite version for mobile

# References

  * YOLOv4: Optimal Speed and Accuracy of Object Detection [YOLOv4](https://arxiv.org/abs/2004.10934).
  * [darknet](https://github.com/AlexeyAB/darknet)
  
   My project is inspired by these previous fantastic YOLOv3 implementations:
  * [Yolov3 tensorflow](https://github.com/YunYang1994/tensorflow-yolov3)
  * [Yolov3 tf2](https://github.com/zzh8829/yolov3-tf2)
