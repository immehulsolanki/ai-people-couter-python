# Project Write-Up

All the tests were performed on LOCAL Setup with following Softwares and hardwares:
### Software compatibility:
1. OpenVino Toolkit V2019.3.379
2. Python V3.6.5 x64
3. TensorFlow 1.15.0 without AVX2
4. OpenCv V4.1.2
5. VisualStudio Community 2019 Version 16.4.2 (Openvino Dependency)
6. VsCode V1.45.1 (Code Editing)

### Hardware Compatibility:
1. OS: Windows* 10 Pro (10.0.18362) X64 (Update as on May 2020)
2. CPU: Intel i7 4790 @2Ghz (Max capacity 3.5Ghz)
3. RAM: 8GB DDR3
4. Storage: SSD 128GB
5. Graphics Driver:  Intel HD Graphics 4600 Driver Version: 20.19.15.4963
                     [ Shader Version: 5.0 ]
                    [ OpenGL* Version: 4.3 ]
                    [ OpenCL* Version: 1.2 ]

## Explaining Custom Layers

- The Intel® Distribution of OpenVINO™ toolkit supports neural network model layers in multiple frameworks including TensorFlow*, Caffe*, MXNet*, Kaldi* and ONYX*

- Almost all the standard Framework's neuralnetwok layers are already defined and configured in to Model optimizer.

- But in case of custom, special trained models and some of non standard public models consist layer topology that can’t be recognozed by Model Optimizer called as Custome layers.

- In other words Cutom layers are which not listed in to bydefault in model optimizer.
Before converting model in to Intermediate representation any custome layers must be configured properly.

- Device specific plugin will first search for known/prelisted layers. If there are any custome layers it will throw an error.

- To registor or configure custom layers, model optimizer starts with a library of known as "extractors and operations" for each supported model frameworks.

- Custom Layer Extractor:
Responsible for identifying the custom layer operation and extracting the parameters for each instance of the custom layer.

- Custom Layer Operation:
Responsible for specifying the attributes that are supported by the custom layer and computing the output shape for each instance of the custom layer from its parameters.

- Each device plugin includes a library of optimized implementations to execute known layer operations which must be extended to execute a custom layer. The custom layer extension must be implemented according to the target device: Like CPU or GPU

- And finally it will generate a shared library extension according to device. Custom Layer CPU Extension consist a compiled shared library (.so or .dll binary) 

- Custom Layer GPU Extension
OpenCL source code (.cl) for the custom layer kernel that will be compiled to execute on the GPU along with a layer description file (.xml).

- The models utilized in this project were standared TensorFlow models and was fully supported by model optimizer.
There is no custom layers processing are required.

- Reference: https://docs.openvinotoolkit.org

## Comparing Model Performance

Following tests were performed on the sample video located in ./resources/io/ folder of this project. [People counter Video with 10FPS](./resources/io/pca10.mp4)

My method(s) to compare models before and after conversion to Intermediate Representations:
1. Impact on size
2. Impact on CPU, RAM and Network

Additionally, Models were tested in following condition:
1. Accuracy in Counting
2. Accuracy in Duration 
3. Error in Detection of mutiboundingboxes
4. Fluctuation in Video frames
5. Change in Video FPS and resolution.
6. Actual inference time 
7. Inference time delay impact on calculating duration
8. Impact of Video FPS and calculation of stats.
9. Confidence threshold effect on Accuracy of stats.
10. OpenCv waitkey delay effect on processing FPS and Stats.

### Fluctuation in Video frames:
- No model can detect objects with 100% of accuracy in each frames, so there will be some frames that individually skips the detection and may effect on the over all counting of persons.
- In this project this immidiate fluctuation of detection is hadle by two approches: Detectiong and counting if person appears between given time window like 1000ms, any lesser value it wont count, and second approch is FPS base, with respect to FPS of input video count increases and ignpored the values if occured any lesser frames than input FPS.

### Effect of Change in Video FPS:

Video fps will direct impact on calcuating the duration of person, here are the different FPS and its impact on Stats:
- Test Model: person-detection-retail-0013.xml | FP32

FPS: 10
- No Of person: [1, 2, 3, 4, 5, 6]
- Duration: [12.7, 21.4, 18.1, 11.6, 26.1, 11.1]
- Frame No.: [191, 445, 690, 865, 1188, 1356]

FPS: 30
- No Of person: [1, 2, 3, 4, 5, 6]
- Duration: [12.9, 21.4, 18.1, 11.6, 26.1, 11.3]
- Frame No.: [577, 1333, 2068, 2593, 3562, 4072]

FPS: 60
- No Of person: [1, 2, 3, 4, 5, 6]
- Duration: [12.9, 21.4, 18.1, 11.6, 26.1, 11.1]
- Frame No.: [1153, 2665, 4135, 5185, 7123, 8131]

It is clearly seen that change in FPS do effect on stats, but this change is negligible.

### Inference time delay impact on calculating duration:

When we process video on live feed, it will slow down processing because of added delay of inference time, and thus it will change any time related stats in video, It is fixed by either adjusting inference time delay with actual time or counting stats with respect to FPS, in this project both approches are implemented.

### Confidence threshold effect on Accuracy of stats:
Confidence threshold is most important factor, as it will provide the final result of processing from models, so it is recommanded to check the appropriate value by manual trial and error method before deploying an app. 

### OpenCv waitkey delay effect on processing FPS and Stats.
Opencv provides all the imageprocessing facility, but it also has its own gui module called high gui, whenever we call the imshow fuction it will generate the new window for it. And while in active window of opencv it has its own keyboard inturpt method to terminate program called waitkey(ms). This fuction will scan the input keyboard key between provided delays in ms, thus it will also slow down the whole program, so it is also necessory to adjust time of calculated stats accordingly.

### Performance report:
**Abbreviations:**
- *TF = TensorFlow

- *dldt = Intel deep learning deployment toolkit

- *FP = Floating Point
- Without OpenVino = Code on TensorFlow framework

**Table 1: Given Model ID, Name and link**
| ID    | Model Name    |  FrameWork |
| ----- | ------------- | --------- |
| 1 | [faster_rcnn_inception_v2_coco_2018_01_28](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)    | TF
| 2 | [faster_rcnn_nas_coco_2018_01_28](http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_coco_2018_01_28.tar.gz)  | TF
| 3 | [faster_rcnn_resnet101_lowproposals_coco_2018_01_28](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_lowproposals_coco_2018_01_28.tar.gz)    | TF
| 4 | [ssd_inception_v2_coco_2018_01_28](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)    | TF
| 5 | [ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.tar.gz)    | TF
| 6 | [ssdlite_mobilenet_v2_coco_2018_05_09](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz)    | TF
| 7 | [Intel Pre-Trained person-detection-retail-0013](https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html)    | dldt

Tensorflow Model Zoo: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

**Model Download and Intel IR conversion parameters [Linux]:**

Download the model with:
```
sudo wget "model_url"
```

Extract the files:

```
tar -xvf modelname.tar.gz
```

On linux (udacity workspace), the path of model converter is as follows:

```
/opt/intel/openvino/deployment_tools/model_optimizer/mo.py
```
Support files path, Select the .JSON file with model's type:

```
/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
```

So, adjust paths accordingly on following windows commands.

Sample (Assumed model located in /home/workspace/ssd_mobilenet_v2_coco_2018_03_29/ dir.):
```
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model /home/workspace/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config /home/workspace/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json

```

**Model Download and Intel IR conversion parameters [Windows10]:**

- Downloading any model is straight forward, just click on the above links and save it to ./resources/model/ folder and extract the file.
- Extract the model files using winrar or 7zip program.
- Now Open command prompt in **downloaded model folder**.

1.faster_rcnn_inception_v2_coco_2018_01_28

- FP32
```
python "C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\deployment_tools\model_optimizer\mo.py" --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --tensorflow_use_custom_operations_config  "C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\deployment_tools\model_optimizer\extensions\front\tf\faster_rcnn_support.json" --input_shape=[1,600,1024,3] --reverse_input_channels --input=image_tensor
```

- FP16
```
python "C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\deployment_tools\model_optimizer\mo.py" --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --tensorflow_use_custom_operations_config  "C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\deployment_tools\model_optimizer\extensions\front\tf\faster_rcnn_support.json" --input_shape=[1,600,1024,3] --reverse_input_channels --input=image_tensor --data_type FP16
```

2.faster_rcnn_nas_coco_2018_01_28

- FP32
```
python "C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\deployment_tools\model_optimizer\mo.py" --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --tensorflow_use_custom_operations_config  "C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\deployment_tools\model_optimizer\extensions\front\tf\faster_rcnn_support.json" --input_shape=[1,1200,1200,3] --reverse_input_channels --input=image_tensor
```

- FP16
```
python "C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\deployment_tools\model_optimizer\mo.py" --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --tensorflow_use_custom_operations_config  "C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\deployment_tools\model_optimizer\extensions\front\tf\faster_rcnn_support.json" --input_shape=[1,1200,1200,3] --reverse_input_channels --input=image_tensor --data_type FP16
```

3.faster_rcnn_resnet101_lowproposals_coco_2018_01_28

- FP32
```
python "C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\deployment_tools\model_optimizer\mo.py" --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --tensorflow_use_custom_operations_config  "C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\deployment_tools\model_optimizer\extensions\front\tf\faster_rcnn_support.json" --input_shape=[1,600,1024,3] --reverse_input_channels --input=image_tensor
```

- FP16
```
python "C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\deployment_tools\model_optimizer\mo.py" --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --tensorflow_use_custom_operations_config  "C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\deployment_tools\model_optimizer\extensions\front\tf\faster_rcnn_support.json" --input_shape=[1,600,1024,3] --reverse_input_channels --input=image_tensor --data_type FP16
```

4.ssd_inception_v2_coco_2018_01_28

- FP32
```
python "C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\deployment_tools\model_optimizer\mo_tf.py" --input_model frozen_inference_graph.pb --reverse_input_channels --input=image_tensor --tensorflow_object_detection_api_pipeline_config pipeline.config --tensorflow_use_custom_operations_config ssd_v2_support.json --input_shape=[1,300,300,3]
```

- FP16
```
python "C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\deployment_tools\model_optimizer\mo_tf.py" --input_model frozen_inference_graph.pb --reverse_input_channels --input=image_tensor --tensorflow_object_detection_api_pipeline_config pipeline.config --tensorflow_use_custom_operations_config ssd_v2_support.json --input_shape=[1,300,300,3] --data_type FP16
```

5.ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03

- FP32
```
python "C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\deployment_tools\model_optimizer\mo_tf.py" --input_model frozen_inference_graph.pb --reverse_input_channels --input=image_tensor --tensorflow_object_detection_api_pipeline_config pipeline.config --tensorflow_use_custom_operations_config ssd_v2_support.json --input_shape=[1,300,300,3]
```

- FP16
```
python "C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\deployment_tools\model_optimizer\mo_tf.py" --input_model frozen_inference_graph.pb --reverse_input_channels --input=image_tensor --tensorflow_object_detection_api_pipeline_config pipeline.config --tensorflow_use_custom_operations_config ssd_v2_support.json --input_shape=[1,300,300,3] --data_type FP16
```

6.ssdlite_mobilenet_v2_coco_2018_05_09

- FP32
```
python "C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\deployment_tools\model_optimizer\mo_tf.py" --input_model frozen_inference_graph.pb --reverse_input_channels --input=image_tensor --tensorflow_object_detection_api_pipeline_config pipeline.config --tensorflow_use_custom_operations_config ssd_v2_support.json --input_shape=[1,300,300,3]
```

- FP16
```
python "C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\deployment_tools\model_optimizer\mo_tf.py" --input_model frozen_inference_graph.pb --reverse_input_channels --input=image_tensor --tensorflow_object_detection_api_pipeline_config pipeline.config --tensorflow_use_custom_operations_config ssd_v2_support.json --input_shape=[1,300,300,3] --data_type FP16
```
7.person-detection-retail-0013

- FP32, FP16, INT8
```
$inteldir/downloader.py --name person-detection-retail-0013 -o /$outputdir
```

**Table 2: RAM Utilization:**
| ID    | FP    | Without OpenVino (MB) | With OpenVino (MB)    | Difference (%)(-MB)| 
| ----- | ----- | --------------------- | --------------------- | ------------------ |     
| 1 | FP32  | 670   | 236   | 64% (434)| 
| 1 | FP16  | NA    | 203   | 69.7% (467)| 
| 2 | FP32  | 3170  | <font color="red">Failed Out of memory</font> | NA
| 2 | FP16  | NA    | <font color="red">Failed Out of memory</font>     | NA
| 3 | FP32  | 1290  | 471   | 63.4% (819)
| 3 | FP16  | NA    | 403   | 68.7% (887)
| 4 | FP32  | 631   | 327   | 48.1% (304)
| 4 | FP16  | NA    | 189   | 70.04% (442)
| 5 | FP32  | 392   | 120   | 69.3% (272)
| 5 | FP16  | NA    | 124   | 68.36% (268)
| 6 | FP32  | 321   | 144   | 55.14% (177)
| 6 | FP16  | NA    | 126   | 60.74% (195)

**Table 3: CPU Utilization (Approximately +- 3% ):**
| ID    | FP    | Without OpenVino (%)  | With OpenVino (%) | Difference(%) (-value)| 
| ----- | ----- | --------------------- | ----------------- | --------------------- | 
| 1 | FP32  | 50%   | 45%   | 10% (5)| 
| 1 | FP16  | NA    | 43%   | 14% (7)| 
| 2 | FP32  | 58%   | <font color="red">Failed Out of memory</font> | NA| 
| 2 | FP16  | NA    | <font color="red">Failed Out of memory</font> | NA| 
| 3 | FP32  | 55%   | 55%   | 0%| 
| 3 | FP16  | NA    | 46%   | 16.3% (9)| 
| 4 | FP32  | 46%   | 45%   | 2.1% (1)| 
| 4 | FP16  | NA    | 45%   | 2.1% (1)| 
| 5 | FP32  | 40%   | 39%   | 2.5% (1)| 
| 5 | FP16  | NA    | 41%   | <font color="red">2.5%(+1)</font>| 
| 6 | FP32  | 43%   | 40%   | 6.9% (3)| 
| 6 | FP16  | NA    | 40%   | 6.9% (3)| 

**Table 4: Inference Time:**
| ID    | FP    | Without OpenVino (ms) | With OpenVino (ms)  | Difference (-ms) |
| ----- | ----- | -------------------- | ------------------ | ---------------- | 
|  |  | **Infer. time: [min max avg.]**  | **Infer. time: [min max avg.]** |  **In Avg. time** | 
| 1 | FP32  | [765.55, 7456.59, 830.49]       | [453.08, 703.07, 622.33]    | 25.06% (208.16) | 
| 1 | FP16  | NA                              | [437.41, 703.08, 595.9] | 28.24% (234.59)| 
| 2 | FP32  | [35263.16, 68917.08, 41501.58]  | <font color="red">Failed Out of memory</font>   | NA | 
| 2 | FP16  | NA                              | <font color="red">Failed Out of memory</font>   | NA | 
| 3 | FP32  | [1609.25, 15389.54, 1792.6]     | [1031.16, 1562.39, 1267.09] | 29.31%  (525.51) | 
| 3 | FP16  | NA                              | [1031.16, 1609.28, 1349.45] | 24.72%  (443.15) | 
| 4 | FP32  | [109.35, 7655.72, 141.96]       | [78.1, 147.09, 82.4]    | 41.95% (59.56) | 
| 4 | FP16  | NA                              | [46.82, 109.37, 53.47]  | 62.33% (88.49) | 
| 5 | FP32  | [46.86, 3765.31, 58.88]         | [15.61, 31.27, 16.84]   | 71.39% (42.04) | 
| 5 | FP16  | NA                              | [15.62, 31.26, 16.87]   | 71.34% (42.01) | 
| 6 | FP32  | [62.48, 3783.11, 69.39]         | [15.61, 31.27, 17.34]   | 75.01% (52.05) | 
| 6 | FP16  | NA                              |  [15.61, 31.26, 17.59]    | 74.65% (51.8) | 

**Table 5: Size Comparison:**
| ID    | FP    | Before [MB]   | After (MB) IR+BIN | Difference% (-MB)| 
| ----- | ----- | ------------- | ----------------- | ---------------- | 
| 1 | FP32  | 55.8  | 50.8  | 8.9% (5MB) | 
| 1 | FP16  | NA        | 25.5  | 54.3% (30MB) | 
| 2 | FP32  | 414.6 | 401       | 3.2% (13MB) | 
| 2 | FP16  | NA        | 201       | 49.8% (200MB) | 
| 3 | FP32  | 191.8 | 183       | 4.1% (8MB) | 
| 3 | FP16  | NA        | 91.9  | 51.8% (99.1MB) | 
| 4 | FP32  | 99.5  | 95.5  | 4% (4MB) | 
| 4 | FP16  | NA        | 47.8  | 51.9% (51.7MB) | 
| 5 | FP32  | 10.5  | 13.3  | <font color="red">+26.6% (+2.8MB)</font> | 
| 5 | FP16  | NA        | 6.7       | 36.1% (3.8MB) | 
| 6 | FP32  | 19.4  | 17.1  | 11.8% (2.3MB) | 
| 6 | FP16  | NA        | 8.6       | 55.6% (10.8MB) | 

**Accuracy (Counting, Detection, and Error):**

**1. faster_rcnn_inception_v2_coco_2018_01_28 :**

Status: <font color="green">Success</font>

This model provides very good accuracy in detection and counting. On tensorflow code this model runs with 0.5 confidence threshold successfully. However, there are some multiple detection errors. While, on OpenVino, because of optimization and reduction in accuracy it runs with 0.95 confidence threshold successfully. In case of good hardware availability, this model can be used to deploy an Application. But in case of IoT devices, due to high inference time this model may not be useful.

**Table 6:**
| Type                     | Without OpenVino                     | With OpenVino| 
| ------------------------ | ------------------------------------ | ------------ | 
| FP32                     | No Of person:                        | No Of person: | 
| Confidence               | [1, 2, 3, 4, 5, 6]                   | [1, 2, 3, 4, 5, 6] | 
| Threshold:               | Duration:                            | Duration: | 
| 0.5                      | [13.7, 22.0, 19.4, 12.2, 27.7, 12.2] | [12.7, 21.5, 18.7, 11.6, 25.4, 11.7] | 
|                          | Error:                               | Error: | 
|                          | Frame No: Count                      | N/A | 
|                          | [['F: 196 C: 2'], ['F: 696 C: 2'],   |  | 
|                          | ['F: 1190 C: 2'], ['F: 1197 C: 2'],  |  | 
|                          | ['F: 1352 C: 2'], ['F: 1353 C: 2']]  |  | 
|                          |                                      |  | 
| FP16                     | NA                                   | No Of person: | 
| Confidence               |                                      | [1, 2, 3, 4, 5, 6] | 
| Threshold:               |                                      | Duration: | 
| 0.95                     |                                      | [12.9, 21.6, 18.9, 12.1, 26.5, 12.0] | 
|                          |                                      | Error: | 
|                          |                                      | N/A | 

    
**2. faster_rcnn_nas_coco_2018_01_28:**

Status: <font color="red">Failed</font>

This models provides highest accuracy in detection, on tensor flow code it takes approximately 40Sec time to process each frame, which is not good for IoT devices with hardware limitations. While on OpenVino, Model fails to load and throws a memory error.

**3. faster_rcnn_resnet101_lowproposals_coco_2018_01_28**

Status: <font color="green">Success</font>

This model accuracy is moderate. On tensor flow code it runs with confidence threshold 0.5 successfully, while on OpenVino it runs with confidence 0.9 due to loss in accuracy during conversion. Although there are some multiple detection occurred, but one or two frame differences are taken case in filter in program. 
With good resources this model can be deployed on edge. But for IoT, because of high inference time, model cannot be usefull.

**Table 7:**
| Type                     | Without OpenVino                     | With OpenVino| 
| ------------------------ | ------------------------------------ | ------------ | 
| FP32                     | Confidence: 0.5                      | Confidence: 0.9 | 
|                          | No Of person:                        | No Of person: | 
|                          | [1, 2, 3, 4, 5, 6]                   | [1, 2, 3, 4, 5, 6] | 
|                          | Duration:                            | Duration: | 
|                          | [13.6, 22.0, 19.6, 12.0, 27.4, 12.2] | [13.1, 21.7, 19.1, 11.9, 26.9, 12.2] | 
|                          | Error:                               | Error: | 
|                          | N/A                                  | Frame No: Count | 
|                          |                                      | [['F: 186 C: 2'], ['F: 1178 C: 2'],  | 
|                          |                                      | ['F: 1184 C: 2']] | 
|                          |                                      |  | 
| FP16                     | NA                                   | No Of person: | 
|                          |                                      | [1, 2, 3, 4, 5, 6] | 
|                          |                                      | Duration: | 
|                          |                                      | [13.1, 21.7, 19.1, 11.9, 26.9, 12.2] | 
|                          |                                      | Error: | 
|                          |                                      | Frame No: Count | 
|                          |                                      | [['F: 186 C: 2'], ['F: 1178 C: 2'],  | 
|                          |                                      | ['F: 1184 C: 2']] | 


**4. ssd_inception_v2_coco_2018_01_28**

Status: <font color="red">Failed</font>

This model is has good detection accuracy and inference time on tensorflow code, but after conversion, on OpenVino it fails to count person and duration at confidence 0.1 due to reduction in accuracy.

**Table 8:**
| Type                     | Without OpenVino                     | With OpenVino| 
| ------------------------ | ------------------------------------ | ------------ | 
| FP32                     | No Of person:                        |<font color="red">Failed</font> | 
| Confidence               | [1, 2, 3, 4, 5, 6]                   |  | 
| Threshold:               | Duration:                            |  | 
| 0.3                      | [10.3, 11.5, 17.6, 11.9, 19.9, 12.2] |  | 
|                          | Error:                               |  | 
|                          | N/A                      |  | 
|                          |                                      |  | 
| FP16                     | NA                                   | <font color="red">Failed</font> | 
| Confidence               |                                      |  | 
| Threshold:               |                                      |  | 
| NA                       |                                      |  | 

**5. ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03**

Status: <font color="red">Failed</font>

This model has very low detection accuracy by default, and it fails even on TensorFlow code, at confidence 0.49 it misses the count and at confidence 0.5, multiple detection increases. Thus not compatible to deploy an app. 

**Table 9:**
| Type                     | Without OpenVino                     | With OpenVino| 
| ------------------------ | ------------------------------------ | ------------ | 
| FP32                     |<font color="red">Failed</font>       |<font color="red">Failed</font> | 
| Confidence               |                                      |  | 
| Threshold:               |                                      |  | 
| 0.49 & 0.5               |                                      |  | 
|                          |                                      |  | 
| FP16                     | NA                                   | NA | 
| Confidence               |                                      |  | 
| Threshold:               |                                      |  | 
| NA                       |                                      |  | 

**6. ssdlite_mobilenet_v2_coco_2018_05_09**

Status: <font color="red">Failed</font>

This model successfully runs on tensor flow code , but after conversion, on OpenVino it fails to count the person due to decrease in accuracy, at confidence 0.3 it misses the person count and at confidence 0.5 it detects multiple boxes and fails the overall counting.
Thus, this models cannot be used deploy an app on OpenVino platform.

**Table 10:**
| Type                     | Without OpenVino                     | With OpenVino| 
| ------------------------ | ------------------------------------ | ------------ | 
| FP32                     | No Of person:                        |Confidence: 0.3 | 
| Confidence               | [1, 2, 3, 4, 5, 6]                   | <font color="red">Failed</font> | 
| Threshold:               | Duration:                            | | 
| 0.3 & 0.5                | [10.7, 9.4, 16.5, 11.9, 22.9, 12.2]  | | 
|                          | Error:                               | | 
|                          | N/A                                  | | 
|                          |                                      |  | 
| FP16                     | NA                                   | <font color="red">Failed</font> | 
| Confidence               |                                      | | 
| Threshold:               |                                      | | 
| NA                       |                                      |  | 

**7. Intel Pre-Trained person-detection-retail-0013**

Status: <font color="green">Success</font>

This model is from Intel open model zoo and pretrained and optimized, it works perfectly and fulfill the edge processing criteria in terms of inference time, performance and accuracy.
This model is perfect for the app and the IoT requirements.

**Table 11:**
| Type       | Stats                                | Utilization                   |
|------------|--------------------------------------|-------------------------------|
| FP32       | No Of person:                        | RAM: 100MB                    |
| Confidence | [1, 2, 3, 4, 5, 6]                   | CPU: 38%                      |
| Threshold: | Duration:                            | Size: 2.90MB                  |
| 0.5        | [12.7, 21.4, 18.1, 11.6, 26.1, 11.1] | Inference time:[min max avg.] |
|            | Error:                               | [12.11, 46.87, 18.2]          |
|            | N/A                                  |                               |
|            |                                      |                               |
| FP16       | No Of person:                        | RAM: 80MB                     |
| Confidence | [1, 2, 3, 4, 5, 6]                   | CPU: 38%                      |
| Threshold: | Duration:                            | Size: 1.52MB                  |
| 0.5        | [12.7, 21.4, 18.1, 11.6, 26.1, 11.1] | Inference time:[min max avg.] |
|            | Error:                               | [15.61, 46.87, 18.11]         |
|            | N/A                                  |                               |
|            |                                      |                               |
| INT 8      | No Of person:                        | RAM: 80MB                     |
| Confidence | [1, 2, 3, 4, 5, 6]                   | CPU: <font color="red">40% </font> |
| Threshold: | Duration:                            | Size: 1.52MB                  |
| 0.5        | [12.7, 21.3, 17.1, 11.6, 25.8, 11.1] | Inference time:[min max avg.] |
|            | Error:                               | <font color="red">[46.85, 141.41, 67.77] </font> |
|            | N/A                                  |                               |


## Assess Model Use Cases

Some of the potential use cases of the people counter app are:
1. Counting people on public places like airport, country borders, railwastations, where security checking is required one person at a time.
2. Super mall queue.
3. Calculation of person duration (average time spent on workplace) in office, school and colledges.
4. Calculation of duration of patient utilized bed in hospitals.
5. Controlling no of people in area, if more people found will give an alert. Like police station, Goverment Offices, Hospital surgery room.
6. Monitoring High alert areas like militory base stations, where continuous monitoring of no. of people is required.

## Assess Effects on End User Needs

**Lighting:**
- It will direct impact on detection of person in video feed, low light condition may give multiple  count and high light condition may miss the count, thus it is important to maintain the lighning conditin according to model requirement and accuracy. 

**Model accuracy:**
- Accuracy of model comes with the computation cost, so it is required to maintain the balance between end user need and model accuracy. Very highly accurate models require high resources and very low acurate model may fail to fulfill the purpose.

**Focal Length and image size:**
- Most of the CCTV cameras does not have dynamic focal length, its either fixed focal lenght or infinit focal length. So in that case infinate focal length give beter result since it does not zoom in or out the actual picture

- In case of public places or open places, where multiple person detection require, it is recommanded to use low focal length camera which capture more filed of view.

- In case of counting person in fixed or controlled area it is required to use high focal length because it capture closer look. Which will increase the accuracy of counting.

**Image size:**
- Since most of the model are trained with fixed and small size of images, thus using high resolution may not impact overall, but if smaller object detection is require then high resolution may give better results. Image size also impact on cpu, high image will require more time to process.


