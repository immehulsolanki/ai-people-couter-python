#!/usr/bin/env python3
"""
Copyright [2020] [MEHUL SOLANKI]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore
import datetime

# Init Log File, will save to current dir with datetime
filenameis = "log_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".txt"
log.basicConfig(filename = filenameis ,level=log.DEBUG)
# log.debug('This message should go to the log file')
# log.info('So should this')
# log.warning('And this, too')

# This code will be reusable in Windows10 and Linux without any change 
class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None
        self.count_input_layers = 0
        self.first_input_layer = None
        self.second_input_layer = None
    
    def load_model(self, model, device="CPU", cpu_extension=None):
        ### TODO: Load the model ###
        self.plugin = IECore()

        log.info("------device avaibility--------")
        # for devices in self.plugin.available_devices: #Dont use device variable, conflicts.
        #     log.info("Available device: "+ str(devices)) #get name of available devices
        log.info("---------Plugin version--------")
        ver = self.plugin.get_versions("CPU")["CPU"] # get plugin info
        log.info("descr: maj.min.num"+ str(ver.description) +"."+ str(ver.major) +"." + str(ver.minor)+"." + str(ver.build_number))

        ### Load IR files into their related class
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        # Check if path is not given or model name doesnt ends with .xml
        if model_xml is not None and model_xml.find('.xml') != -1:
            f,s = model_xml.rsplit(".",1) #check from last "." and "r"split only one element from last
            model_bin = f + ".bin"
        else:
            log.error("Error! Model files are not found or invalid, check paths")
            log.error("Program stopped")
            sys.exit() #exit program no further execution
        log.info("-------------Model path----------")
        log.info("XML: "+ str(model_xml))
        log.info("bin: "+ str(model_bin))

        self.network = IENetwork(model=model_xml, weights=model_bin)
        log.info("ModelFiles are successfully loaded into IENetwork")

        ### TODO: Check for supported layers ###
        log.info("Checking for supported Network layers...")
        # Query network will return all the layer, required all the time if device changes.
        supported_layers = self.plugin.query_network(network=self.network, device_name="CPU")
        log.info("------Status of default Network layers--------")
        log.info("No. of Layers in network"+ str(len(self.network.layers)))
        log.info("No. of supported layers:"+ str(len(supported_layers)))

        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            log.info("Unsupported layers found:"+ str(unsupported_layers))
            log.info("CPU extension required and adding...")
            #exit(1)
        ### TODO: Add any necessary extensions ###
            if cpu_extension and "CPU" in device:
                self.plugin.add_extension(cpu_extension, device)
                log.info("Checking for CPU extension compatibility...")
                # Again Query network will return fresh list of supported layers.
                supported_layers = self.plugin.query_network(network=self.network, device_name="CPU")
                log.info("------Status of Network layers with CPU Extension--------")
                log.info("No. of Layers in network"+ str(len(self.network.layers)))
                log.info("No. of supported layers:"+ str(len(supported_layers)))

                unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
                if len(unsupported_layers) != 0:
                    log.error("Unsupported layers found: "+ str((unsupported_layers)))
                    log.error("Error! Model not supported, Program stopped")
                    exit()
            else:
                log.error("Error! cpu extension not found")
                log.error("Program stopped")
                exit()
        else:
            log.info("All the layers are supported, No CPU extension required")
        
        ### TODO: Return the loaded inference plugin ###
        # This will enable all following four functions ref:intel doc. ie_api.IECore Class Reference
        self.exec_network = self.plugin.load_network(self.network, "CPU")
        log.info("IR successfully loaded into Inference Engine")

        ### Note: You may need to update the function parameters. ###
        return

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        # Get the input layer

        # To avoid unnecessory conversion in realtime in exec_net() and get_input_shape()
        self.count_input_layers = len(list(self.network.inputs))

        log.info("-----Accessing input layer information-----")
        log.info('Network input layers = ' + str(list(self.network.inputs)))
        log.info('Network input layers type: '+ str(type(self.network.inputs)))

        # FasterRcnn model Fix for Openvino V2019R3
        if self.count_input_layers > 1: # check if more than 1 input layers
            # Model:TF-faster_rcnn_inception_v2_coco_2018_01_28
            # Network input layers name = ['image_info', 'image_tensor']
            # Access it by dictionary element

            # To avoid unnecessory conversion in realtime in exec_net()
            self.first_input_layer = list(self.network.inputs)[0]
            self.second_input_layer = list(self.network.inputs)[1]
            #print("Var:",self.first_input_layer,self.second_input_layer) #Debug

            log.warning("More than one input layers found")
            log.warning("### Warning!!! Manual data feed may require... ###")
            log.warning("Read respective model documentation for more info.")
            self.input_blob =  self.network.inputs[self.second_input_layer]
            log.warning("Manually selected input layer: "+ str(self.second_input_layer))
            log.warning("Fixed data applied to other input layer: "+ str(self.first_input_layer) +":" + str([600,1024,1]))
            log.warning("in function exec_net()")
            log.info("-------------------------------")
            return self.input_blob.shape
        else: # If regular model 
            self.input_blob = next(iter(self.network.inputs))#Origional
            log.info("-------------------------------")
            return self.network.inputs[self.input_blob].shape #Origional

    def exec_net(self,image):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        
        # FasterRcnn model Fix for Openvino V2019R3
        if self.count_input_layers > 1: # check if more than 1 input layers
            # Model:TF-faster_rcnn_inception_v2_coco_2018_01_28
            # Network input layers name = ['image_info', 'image_tensor']
            # Access it by dictionary element
            # Manually feed data according to model documentation
            # inputs layes names={'image_info':[600,1024,1],'image_tensor': image})
            # image_info:[HxWxS], height, width and scale factor usually 1
            self.exec_network.start_async(request_id=0, 
                inputs={self.first_input_layer:[600,1024,1],self.second_input_layer: image})      
        else: # If regular model
            self.exec_network.start_async(request_id=0, #Origional
                inputs={self.input_blob: image})
        return

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        status = self.exec_network.requests[0].wait(-1)
        return status

    def get_output(self):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        self.output_blob = next(iter(self.network.outputs))
        # First return the name of blob as dictionary and second output of first blob as Nd array
        return self.exec_network.requests[0].outputs, self.exec_network.requests[0].outputs[self.output_blob]
