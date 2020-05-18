"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
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


import os
import sys
import time
import socket
import json
import cv2
import numpy as np
import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

#Win10 CPU_EXTENSION Path Openvino V2019R3
CPU_EXTENSION = r"C:/Program Files (x86)/IntelSWTools/openvino_2019.3.379/deployment_tools/inference_engine/bin/intel64/Release/cpu_extension_avx2.dll"
#Linux CPU_EXTENSION Path Openvino V2019R3
#CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
BOXCOLOR = {'RED':(0,0,255),'GREEN':(0,255,0),'BLUE':(255,0,0),'WHITE':(255,255,255),'BLACK':(0,0,0)}

#MQTT server environment variables
# HOSTNAME = socket.gethostname()
# IPADDRESS = socket.gethostbyname(HOSTNAME)
# MQTT_HOST = IPADDRESS
# MQTT_PORT = 1883 #Udacity port 3001
# MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image, video file or for webcam just type CAM")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=CPU_EXTENSION,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-c", "--box_color", type=str, default="WHITE",
                        help="Color of bounding box[RED,GREEN,BLUE,WHITE,RED]"
                        "(WHITE by default)")
    parser.add_argument("-ap", "--alarm_people", type=int, default=1,
                        help="Alarm when certain no people detected exceed the limit"
                        "(1 by default)")
    parser.add_argument("-ad", "--alarm_duration", type=int, default=15,
                        help="Alarm when time of person stayed exceed the limit"
                        "(15sec. by default)")
    parser.add_argument("-tvf", "--toggle_video", type=str, default="ON",
                        help="Toggle Video feed on or off [ON or OFF]"
                        "(on by default)")
    parser.add_argument("-ci", "--cam_id", type=int, default=0,
                        help="input web Camera id"
                        "(0 by default)")
    parser.add_argument("-db", "--delay_band", type=int, default=1000,
                        help="input delay band (Millis) to fix counting in case of video fluctuation or frame loss"
                        "(1000 millis by default)")
                        
    return parser


# def connect_mqtt():
#     ### TODO: Connect to the MQTT client ###
#     #client = None
#     client = mqtt.Client() #Fixed Syntax
#     client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL) # Syntax with parameter of ip port 
#     return client

def check_input_type(input, id):
    """
    check input is video,image or cam
    """
    #print("input",input)
    checkInputargs = input #string from args.input
    checkError = checkInputargs.find(".") #Verify If there is extension or other than CAM
    error_flag = False
    image_flag = False
    cap = None
    if checkInputargs == "CAM": #check for cam
        cap = cv2.VideoCapture(id) #Assign CAM ID
        print("Performing inference on webcam video...")
    elif checkError is -1:  #Check for if there any  extension
        print("Error: invalid input or currupted file") #error for no extension
        print("Use -h argument for help")
        error_flag = True
    else:
        path,ext= checkInputargs.rsplit(".",1) #find extension
        if ext == "bmp" or ext == "jpg": #supporeted ext.
            print("Performing inference on single image...")
            cap = cv2.VideoCapture(input)
            image_flag = True
        elif ext == "mp4" or ext == "MP4": #if not image feed video
            cap = cv2.VideoCapture(input) #Load local stream
            print("Performing inference on local video...")
        else:
            print("Image/Video formate not supported")
            error_flag = True
    return cap, error_flag, image_flag


def preprocess_frame(frame,height,width):
    p_frame = cv2.resize(frame, (height, width)) #Resize as per network input spec.
    p_frame = p_frame.transpose((2,0,1)) #swap channel cxhxw 
    p_frame = p_frame.reshape(1, *p_frame.shape) #add one axis 1 to make 4D shape for network input
    #print(p_frame.shape) #Debug output
    return p_frame

def draw_boxes(frame, result, width, height, color, prob_threshold):
    '''
    Draw bounding boxes onto the frame.
    '''
    countBox = 0
    countmultipeople = 0
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= prob_threshold:
            countBox = 1
            countmultipeople += 1
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            label = "Person"+str(countmultipeople)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 1) #main rect.
            cv2.rectangle(frame, (xmin, ymin), (xmin+90, ymin+10), color, -1) # Text rect.
            cv2.putText(frame, label, (xmin,ymin+10),cv2.FONT_HERSHEY_PLAIN, 0.8, BOXCOLOR['BLACK'], 1)
    return frame, countBox, countmultipeople

def draw_framelinegreen(frame,height,width): #Better to pass Color parameter
    """
    Draw normal Green frame on video
    """
    # Draw line top left and right
    cv2.line(frame, (0, 0), (0,int(height/10)), (0,255,0),10)#line top teft horizontal.
    cv2.line(frame, (0, 0), (int(height/10),0), (0,255,0),10)#line top left vertical.
    cv2.line(frame, (width, 0), (width-int(height/10),0), (0,255,0),10)#line top right horizontal.
    cv2.line(frame, (width, 0), (width,int(height/10)), (0,255,0),10)#line top right vertical. 

    # Draw line bottom left and right
    cv2.line(frame, (0, height), (0,height-int(height/10)), (0,255,0),10)#line.
    cv2.line(frame, (0, height), (int(height/10),height), (0,255,0),10)#line.
    cv2.line(frame, (width, height), (width-int(height/10),height), (0,255,0),10)#line.
    cv2.line(frame, (width, height), (width,height-int(height/10)), (0,255,0),10)#line.
    return frame

def draw_framelinered(frame,height,width): #Better to pass Color parameter
    """
    Draw alert red frame on video
    """
    # Draw line top left and right
    cv2.line(frame, (0, 0), (0,int(height/10)), (0,0,255),10)#line top teft horizontal.
    cv2.line(frame, (0, 0), (int(height/10),0), (0,0,255),10)#line top left vertical.
    cv2.line(frame, (width, 0), (width-int(height/10),0), (0,0,255),10)#line top right horizontal.
    cv2.line(frame, (width, 0), (width,int(height/10)), (0,0,255),10)#line top right vertical. 

    # Draw line bottom left and right
    cv2.line(frame, (0, height), (0,height-int(height/10)), (0,0,255),10)#line.
    cv2.line(frame, (0, height), (int(height/10),height), (0,0,255),10)#line.
    cv2.line(frame, (width, height), (width-int(height/10),height), (0,0,255),10)#line.
    cv2.line(frame, (width, height), (width,height-int(height/10)), (0,0,255),10)#line.
    return frame

def selectBoxcolor(color):
    """
    To change bounding box color
    """
    if color == 'RED':
        color = BOXCOLOR['RED']
    elif color == 'GREEN':
        color = BOXCOLOR['GREEN']
    elif color == 'BLUE':
        color = BOXCOLOR['BLUE']
    elif color == 'WHITE':
        color = BOXCOLOR['WHITE']
    elif color == 'BLACK':
        color = BOXCOLOR['BLACK']
    #print("Color Selected:",color)
    return color

#def infer_on_stream(args, client): << Origional
def infer_on_stream(args):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    # ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()

    print("Selected Network input Layer type is",type(net_input_shape),"And shape is",net_input_shape)
    print("Required input img size W",net_input_shape[3],"H",net_input_shape[2])

    # ### TODO: Handle the input stream ###
    #cap = cv2.VideoCapture(args.input)
    cap, error_flag, image_flag = check_input_type(args.input, args.cam_id) #call function
    #print("Cap debug",cap, error_flag, image_flag) #debug return
    if error_flag: # Check for invalid file extension
        print("Program stopped")
        return
    elif image_flag: #check for image 
        INPUT_IMAGE = args.input
        img = cv2.imread(INPUT_IMAGE)
        if (type(img) is not np.ndarray):  #check for if image read empty same as img.empty()
            print("Error: Invalid image or path")
            print("Use -h argument for help")
            return
    else:
        cap.open(args.input)

    # Get input feed height and width
    img_width = int(cap.get(3))
    img_height = int(cap.get(4))
    print("feed frame size W",img_width,"H",img_height)

    # Initialize video writer if video mode
    if not image_flag:
        # Video writer Windows10
        print("---Opencv video writer debug WIN---")
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter('out.mp4', fourcc, 10, (img_width,img_height))
        print("-------------------------------")
        # Video writer Linux
        # print("---Opencv video writer debug LIN---")
        # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        # out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (img_width,img_height))
        # print("-------------------------------")

    # Initialized varible utilized inside loop
    frame_count = 0 
    total_people_count = 0
    last_state = 0
    delay_on = 0
    delay_off = (time.time() * 1000) # Initialize timer before loop to get actual time
    delay_diff_on = 0
    delay_diff_off = 0
    duration = 0
    count_people_image = 0

    # Second counting timer initialized
    sec_on = (time.time() * 1000) # Timer for update stat on terminal START
    sec_diff = 0
    cv_drawstate_time_s = 0
    cv_drawstate_time_e = 0
    count_flag = False

    # Accuracy Log
    Accuracy_log = {}
    acount = 0
    error_log = {'MuliBoxDetected':{}}
    ecount = 0 #counter for error log in case of multiple box count


    # ### TODO: Loop until stream is over ###
    while cap.isOpened():
        frame_count += 1 # Count no of frame processed.
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        ### TODO: Read from the video capture ###
        ### TODO: Pre-process the image as needed ###
        p_frame = preprocess_frame(frame,net_input_shape[3],net_input_shape[2]) #from extracted input function
        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(p_frame)
        ### TODO: Wait for the result ###
        inferreq_start_time = (time.time() * 1000) # Timer for inference START
        if infer_network.wait() == 0:
            inferreq_end_time = (time.time() * 1000) - inferreq_start_time # Timer for inference END
            #print(inferreq_end_time)

            ### TODO: Get the results of the inference request ###
            blob, result = infer_network.get_output()

            # If model outputs multiple blob, print available blob infirmation
            if frame_count == 1: # Print only Once
                for name,output_ in blob.items(): #Find the possible BLOBS for name, 
                    print ("The name of available blob is:", name)

            ### TODO: Extract any desired stats from the results ###
            color = selectBoxcolor(args.box_color)
            cv_drawboxtime_s = (time.time() * 1000) # Timer for drawing box on frame START
            frame, count_box, countmultipeople = draw_boxes(frame, result, img_width, img_height, color, args.prob_threshold)
            cv_drawboxtime_e = (time.time() * 1000) - cv_drawboxtime_s #Timer for drawing box on frame END
            
            count_people_image = countmultipeople # Variable For image stat only 
            ### TODO: Calculate and send relevant information on ###
            if count_box != last_state: #Anythinkg underthis will executed if state changes
                print("I am In")
                acount += 1 # Accuracy Log counter for dynamic key
                if count_box == 1:
                    print("I am in 1")
                    count_flag = True
                    delay_on = (time.time() * 1000)  # Timer for on delay START
                    delay_diff_off = (time.time() * 1000) - delay_off # Timer for off delay END
                    print("diff_off",delay_diff_off)
                    delay_diff_on = 0 # Timer for on delay RESET          
                else:
                    print("I am in 0")
                    count_flag = False
                    delay_diff_on = (time.time() * 1000) - delay_on    # Timer for on delay END
                    delay_off = (time.time() * 1000)  # Timer for off delay START
                    print("diff_on",delay_diff_on) #Debug
                    delay_diff_off = 0 # Timer for off delay RESET

                #print("update on",delay_diff_on) #for debug
                #print("update off",delay_diff_off) #for debug

                if delay_diff_on > args.delay_band:
                    print("count+",delay_diff_on, delay_diff_off)
                    total_people_count += 1 
                    duration = delay_diff_on / 1000 # Convert to Sec.
                    #print("duration:", delay_diff_on / 1000 ,"Sec.") #Debug

                last_state = count_box
                Accuracy_log['Log '+ str(acount)] = ['FrameNo:'+str(frame_count),'CurrentCount:'+
                    str(countmultipeople),'TotalCount:'+str(total_people_count),'duration:'+str(duration)]
                # Debug
                print(['FrameNo:'+str(frame_count),'CurrentCount:'+
                    str(countmultipeople),'TotalCount:'+str(total_people_count),'duration:'+str(duration)])
            else:
                if countmultipeople not in (0,1): #In case of multiple people detected
                    print("Multi count detected:",countmultipeople)
                    ecount += 1
                    # Nested Dictionary one fixed key and second dynamic with input value is list
                    error_log['MuliBoxDetected']['log ' + str(ecount)] = ['FrameNo ',str(frame_count)] 
                    #total_people_count = countmultipeople
                    #print(total_people_count) #debug
        
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            #client.publish("person", json.dumps({"count": countmultipeople, "total":total_people_count}))
            ### Topic "person/duration": key of "duration" ###
            #client.publish("person/duration", json.dumps({"duration": duration}))

        ### This part needed to be optimized
        if args.toggle_video is "ON": # If video feed is off stop unnecessory processing
            cv_drawstate_time_s = (time.time() * 1000) # TImer for draw stat on frame START
            # Draw inference time on image
            label = "Inference time: " + str("{:.2f}".format(inferreq_end_time)) + "ms" #string label
            cv2.putText(frame, label, (15,20),cv2.FONT_HERSHEY_COMPLEX, 0.4, BOXCOLOR['BLUE'], 1)
            label1 = "Total people count: " + str(total_people_count) #string label
            if image_flag or countmultipeople > 1:
                label1 = "Total people count: " + str(count_people_image) #string label
            else:
                label1 = "Total people count: " + str(total_people_count)
            cv2.putText(frame, label1, (15,30),cv2.FONT_HERSHEY_COMPLEX, 0.4, BOXCOLOR['BLUE'], 1)
            if countmultipeople > 1 or image_flag is True:     
                label2 = "Average Time stayed: N/A"     
            else:       
                label2 = "Average Time stayed: " + str("{:.2f}".format(duration)) + "Sec." #string label   
            cv2.putText(frame, label2, (15,40),cv2.FONT_HERSHEY_COMPLEX, 0.4, BOXCOLOR['BLUE'], 1)

            # People count exceed alarm
            if countmultipeople > args.alarm_people or duration > args.alarm_duration:
                draw_framelinered(frame,img_height,img_width)
                if countmultipeople > args.alarm_people:
                    label3 = "Alarm: people count limit exceeded! limit: "+ str(args.alarm_people) #string label  
                    cv2.putText(frame, label3, (15,50),cv2.FONT_HERSHEY_COMPLEX, 0.4, BOXCOLOR['RED'], 1)
                else:
                    label4 = "Alarm: Person stayed longer! limit: " + str(args.alarm_duration) + "Sec."#string label  
                    cv2.putText(frame, label4, (15,60),cv2.FONT_HERSHEY_COMPLEX, 0.4, BOXCOLOR['RED'], 1)
            else:
                draw_framelinegreen(frame,img_height,img_width)
                # Draw cv process time
            label5 = "CV Frame process time: " + str("{:.2f}".format(cv_drawboxtime_e + cv_drawstate_time_e)) + "ms" #string label
            cv2.putText(frame, label5, (15,70),cv2.FONT_HERSHEY_COMPLEX, 0.4, BOXCOLOR['BLUE'], 1)
            cv_drawstate_time_e = (time.time() * 1000) - cv_drawstate_time_s # TImer for draw stat on frame END
        else:
             # Stats of time of cv processing on image frame
            sec_diff = (time.time() * 1000) - sec_on  # Timer for update stat on terminal END
            # print("time in ms: ",sec_diff) # Debug
            if sec_diff > 1000 or sec_diff > 2000: # update stat roughly every 1 sec.
                os.system('cls' if os.name == 'nt' else 'clear')
                print() # Blank print
                print("Video feed is OFF, Terminal will refresh every sec.")
                print("Press ctlr+c to stop execution.")
                # People count on terminal
                if countmultipeople > 1:
                    print("Total people count: ",countmultipeople)
                else: 
                    print("Current people count: ", total_people_count)
                    print("Total people count: ",total_people_count)
                    print("Average Time stayed: ""{:.2f}".format(duration)," Sec.")
                # Alarm on terminal 
                if countmultipeople > args.alarm_people or duration > args.alarm_duration:
                    if countmultipeople > args.alarm_people:
                        print("##### Alarm1 #####")
                        print("People count limit exceeded! limit: "+ str(args.alarm_people))
                        print("##################")
                    else:
                        print("##### Alarm2 #####")
                        print("Person stayed longer! limit: " + str(args.alarm_duration) + "Sec.")#string label 
                        print("##################") 
                print("-----Stats for time -----") 
                print("Inference Time(ms):","{:.2f}".format(inferreq_end_time))
                print("Draw boundingBox time(ms):", "{:.2f}".format(cv_drawboxtime_e))
                print("Draw state time(ms):", "{:.2f}".format(cv_drawstate_time_e))
                print("--------------------------") 
                sec_on = (time.time() * 1000) # Timer for update stat on terminal RESET
                sec_diff = 0 # Timer for update stat on terminal RESET

        # Adjusting timers with inference and cv processing time to fix counting and duration.
        if count_flag:
                #print("before",delay_on)
                delay_on = delay_on + inferreq_end_time + cv_drawboxtime_e + cv_drawstate_time_e
                #print("after",delay_on)
        else:
                delay_off = delay_off + inferreq_end_time + cv_drawboxtime_e + cv_drawstate_time_e

        # Write video or image file
        if not image_flag:
            if args.toggle_video is "ON":
                cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                cv2.imshow('frame',frame)
            #out.write(frame) 
        else:
            cv2.imwrite('output_image.jpg', frame)
            print("Image saved sucessfully!")

        ### TODO: Send the frame to the FFMPEG server ###
        if args.toggle_video is "ON":
            a = None
        ### TODO: Write an output image if `single_image_mode` ###
        if key_pressed == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    #client.disconnect()
    print("Last frame prcessed no: ",frame_count)
    print("-----AccuracyLog-----")
    print(Accuracy_log)
    print("-----Error log-----")
    print(error_log)
    print("-----Finish!------")

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    # This is different method so do not use .m type attributes instead use whole name.
    args = build_argparser().parse_args()

    print("Commandline Arguments received")
    print("-----Information-----")
    print("Model path:",args.model)
    print("Video/Image path:",args.input)
    print("Device:",args.device)
    print("CPU Ext. path:",args.cpu_extension)
    print("BoundingBox color:",args.box_color)
    print("Confidence:",args.prob_threshold)
    print("Alarm People count:",args.alarm_people)
    print("Alarm Person duration Sec.:",args.alarm_duration)
    print("Web cam ID(If any):",args.cam_id)
    print("Delay Band(ms):", args.delay_band)
    print("Toggle video feed on/off:",args.toggle_video)
    print("-----------------------")
    # Connect to the MQTT server
    #client = connect_mqtt()


    # Perform inference on the input stream
    # Infer_on_stream(args, client) <<Origional
    infer_on_stream(args)



if __name__ == '__main__':
    main()
