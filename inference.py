#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore

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

        print("------device avaibility--------",flush=True)
        for devices in self.plugin.available_devices: #Dont use device variable, conflicts.
            print("Available device:",devices) #get name of available devices
        print("---------Plugin version--------",flush=True)
        ver = self.plugin.get_versions("CPU")["CPU"] # get plugin info
        print("{descr}: {maj}.{min}.{num}".format(descr=ver.description, maj=ver.major, min=ver.minor, num=ver.build_number),flush=True)

        ### Load IR files into their related class
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        # Check if path is not given or model name doesnt ends with .xml
        if model_xml is not None and model_xml.find('.xml') != -1:
            f,s = model_xml.rsplit(".",1) #check from last "." and "r"split only one element from last
            model_bin = f + ".bin"
        else:
            print("Error! Model files are not found or invalid, check paths",flush=True)
            print("Program stopped",flush=True)
            sys.exit() #exit program no further execution
        print("-------------Model path----------",flush=True)
        print("XML:",model_xml,flush=True)
        print("bin:",model_bin,flush=True)

        self.network = IENetwork(model=model_xml, weights=model_bin)
        print("ModelFiles are successfully loaded into IENetwork",flush=True)

        ### TODO: Check for supported layers ###
        print("Checking for supported Network layers...",flush=True)
        # Query network will return all the layer, required all the time if device changes.
        supported_layers = self.plugin.query_network(network=self.network, device_name="CPU")
        print("------Status of default Network layers--------",flush=True)
        print("No. of Layers in network",len(self.network.layers),flush=True)
        print("No. of supported layers:",len(supported_layers),flush=True)

        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers),flush=True)
            print("CPU extension required and adding...",flush=True)
            #exit(1)
        ### TODO: Add any necessary extensions ###
            #print(cpu_extension)
            #print(device)
            if cpu_extension and "CPU" in device:
                self.plugin.add_extension(cpu_extension, device)
                print("Checking for CPU extension compatibility...",flush=True)
                # Again Query network will return fresh list of supported layers.
                supported_layers = self.plugin.query_network(network=self.network, device_name="CPU")
                print("------Status of Network layers with CPU Extension--------",flush=True)
                print("No. of Layers in network",len(self.network.layers),flush=True)
                print("No. of supported layers:",len(supported_layers),flush=True)

                unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
                if len(unsupported_layers) != 0:
                    print("Unsupported layers found: {}".format(unsupported_layers),flush=True)
                    print("Error! Model not supported, Program stopped",flush=True)
                    exit()
            else:
                print("Error! cpu extension not found",flush=True)
                print("Program stopped",flush=True)
                exit()
        else:
            print("All the layers are supported, No CPU extension required",flush=True)
        
        ### TODO: Return the loaded inference plugin ###
        # This will enable all following four functions ref:intel doc. ie_api.IECore Class Reference
        self.exec_network = self.plugin.load_network(self.network, "CPU")
        print("IR successfully loaded into Inference Engine",flush=True)

        ### Note: You may need to update the function parameters. ###
        return

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        # Get the input layer

        # To avoid unnecessory conversion in realtime in exec_net() and get_input_shape()
        self.count_input_layers = len(list(self.network.inputs))

        print("-----Accessing input layer information-----",flush=True)
        print('Network input layers = ' + str(list(self.network.inputs)),flush=True)
        print('Network input layers type: ',type(self.network.inputs),flush=True)

        # FasterRcnn model Fix for Openvino V2019R3
        if self.count_input_layers > 1: # check if more than 1 input layers
            # Model:TF-faster_rcnn_inception_v2_coco_2018_01_28
            # Network input layers name = ['image_info', 'image_tensor']
            # Access it by dictionary element

            # To avoid unnecessory conversion in realtime in exec_net()
            self.first_input_layer = list(self.network.inputs)[0]
            self.second_input_layer = list(self.network.inputs)[1]
            #print("Var:",self.first_input_layer,self.second_input_layer) #Debug

            print("More than one input layers found",flush=True)
            print("### Warning!!! Manual data feed may require... ###",flush=True)
            print("Read respective model documentation for more info.",flush=True)
            self.input_blob =  self.network.inputs[self.second_input_layer]
            print("Manually selected input layer: ",self.second_input_layer,flush=True)
            print("Fixed data applied to other input layer: ",self.first_input_layer,":",[600,1024,1],flush=True)
            print("in function exec_net()",flush=True)
            print("-------------------------------",flush=True)
            return self.input_blob.shape
        else: # If regular model 
            self.input_blob = next(iter(self.network.inputs))#Origional
            print("-------------------------------",flush=True)
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
