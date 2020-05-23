""" People Counter. Project Write-Up """
"""
 Copyright (c) 2020 Mehul Solanki.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
# Project Write-Up

All the tests were performed on LOCAL Setup with following Softwares and hardwares:
Software compatibility:
1.OpenVino Toolkit V2019.3.379
2.Python V3.6.5 x64
3.TensorFlow 1.15.0 without AVX2
4.OpenCv V4.1.2
5.VisualStudio Community 2019 Version 16.4.2 (Openvino Dependency)
6.VsCode V1.45.1 (Code Editing)

Hardware Compatibility:
1.OS: Windows* 10 Pro (10.0.18362) X64 (Update as on May 2020)
2.CPU: Intel i7 4790 @2Ghz (Max capacity 3.5Ghz)
3.RAM: 8GB DDR3
4.Storage: SSD 128GB
5.Graphics Driver:  Intel HD Graphics 4600 Driver Version: 20.19.15.4963
                    Shader Version: 5.0
                    OpenGL* Version: 4.3
                    OpenCL* Version: 1.2

## Explaining Custom Layers

The Intel® Distribution of OpenVINO™ toolkit supports neural network model layers in multiple frameworks including TensorFlow*, Caffe*, MXNet*, Kaldi* and ONYX*

Almost all the standard Framework's neuralnetwok layers are already defined and configured in to Model optimizer.

But in case of custom, special trained models and some of non standard public models consist layer topology that cant be recognozed by Model Optimizer called as Custome layers.

In other words Cutom layers are which not listed in to bydefault in model optimizer.
Before converting model in to Intermediate representation any custome layers must be configured properly.

Device specific plugin will first search for known/prelisted layers. If there are any custome layers it will throw an error.

To registor or configure configure custom layers, model optimizer starts with a library of known extractors and operations for each supported model frameworks.

Custom Layer Extractor:
Responsible for identifying the custom layer operation and extracting the parameters for each instance of the custom layer.

Custom Layer Operation:
Responsible for specifying the attributes that are supported by the custom layer and computing the output shape for each instance of the custom layer from its parameters.

Each device plugin includes a library of optimized implementations to execute known layer operations which must be extended to execute a custom layer. The custom layer extension is implemented according to the target device: Like CPU or GPU

And finally it will generate a shared library extension according to device
Custom Layer CPU Extension consist a compiled shared library (.so or .dll binary) 

Custom Layer GPU Extension
OpenCL source code (.cl) for the custom layer kernel that will be compiled to execute on the GPU along with a layer description file (.xml).

The model utilized in this project standared TensorFlow models and was fully supported by model optimizer.
There were no cutome layers processing required.
ref:https://docs.openvinotoolkit.org/

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations:
1. Impact on size
2. Impact on CPU, RAM and Network

So, the Models were tested in following condition:
1. Accuracy in Counting
2. Accuracy in Duration 
3. Error in Detection of mutiboundingbox
4. Fluctuation in Video frames
5. Change in Video FPS and resolution.
6. Actual inference time 
7. Inference time delay impact on calculating duration
8. Impact of Video FPS and calculation of stats.
9. Confidence threshold effect on Accuracy of stats.
10. OpenCv waitekey delay effect on processing FPS and Stats.

Effect of Change in Video FPS:
Video fps will direct impact on calcuating the duration of person, here are the different FPS and its impact on Stats:
Test Model: person-detection-retail-0013.xml | FP32

FPS: 10
No Of person:
[1, 2, 3, 4, 5, 6]
Duration:
[12.7, 21.4, 18.1, 11.6, 26.1, 11.1]
Frame No.:
[191, 445, 690, 865, 1188, 1356]

FPS: 30
No Of person:
[1, 2, 3, 4, 5, 6]
Duration:
[12.9, 21.4, 18.1, 11.6, 26.1, 11.3]
Frame No.:
[577, 1333, 2068, 2593, 3562, 4072]

FPS: 60
No Of person:
[1, 2, 3, 4, 5, 6]
Duration:
[12.9, 21.4, 18.1, 11.6, 26.1, 11.1]
Frame No.:
[1153, 2665, 4135, 5185, 7123, 8131]


## Assess Model Use Cases

Some of the potential use cases of the people counter app are:
1. Counting people on public places like airport, country borders, railwastations, where security checking is required one person at a time.
2. Super mall queue.
3. Calculation of person duration (average time spent on workplace) in office, school and colledges.
4. Calculation of duration of patient utilized bed in hospitals.
5. Controlling no of people in area, if more people found will give an alert. Like police station, Goverment Offices, Hospital surgery room.
6. Monitoring High alert areas like militory base stations, where continuous monitoring of no. of people is required.

## Assess Effects on End User Needs

Lighting:
It will direct impact on detection of person in video feed, low light condition may give multiple  count and high light condition may miss the count, thus it is important to maintain the lighning conditin accordin to model requirement and accuracy. 

Model accuracy:
Accuracy of model comes with the computation cost, so it is required to maintain the balance between end user need and model accuracy. Very highly accurate models require high resources and very low acurate model may fail to fulfill the purpose.

Focal Lenght and image size:
Most of the CCTV cameras does not have dynamic focal length, its either fixed focal lenght or infinit focal length. So in that case infinate focal length give beter result since it does not zoom in or out the actual picture

In case of public places or open places, where multiple person detection require, it is recommanded to use low focal length camera which capture more filed of view.

In case of counting person in fixed or controlled area it is required to use high focal length because it capture more closer look. Which will increase the accuracy of counting.

Image size:
Since most of the model are trained with fixed and small size of images, thus using high resolution may not impact overall, but if smaller object detection is require then high resolution may give better results. Image size also impact on cpu, high image will require more time to process.

