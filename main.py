"""People Counter."""
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
import time
import socket
import json
import cv2
import numpy as np
import logging as log
import paho.mqtt.client as mqtt
import datetime
from argparse import ArgumentParser
from inference import Network


#Win10 CPU_EXTENSION Path Openvino V2019R3
#CPU_EXTENSION = r"C:/Program Files (x86)/IntelSWTools/openvino_2019.3.379/deployment_tools/inference_engine/bin/intel64/Release/cpu_extension_avx2.dll"
#Linux CPU_EXTENSION Path Openvino V2019R3
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
BOXCOLOR = {'RED':(0,0,255),'GREEN':(0,255,0),'BLUE':(255,0,0),'WHITE':(255,255,255),'BLACK':(0,0,0)}

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001 #Udacity port # Default 1883
MQTT_KEEPALIVE_INTERVAL = 60


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
    parser.add_argument("-fps", "--fps", required=True, type=int,
                        help="FPS of Video or webcam, required to get perfect duration calculations.")
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
    parser.add_argument("-tv", "--toggle_video", type=str, default="ON",
                        help="Toggle Video feed on or off [ON or OFF]"
                        "(on by default)")
    parser.add_argument("-ci", "--cam_id", type=int, default=0,
                        help="input web Camera id"
                        "(0 by default)")
    parser.add_argument("-db", "--delay_band", type=int, default=1000,
                        help="input delay band (Millis) to fix counting in case of video fluctuation or frame loss"
                        "(1000 millis by default)")
    parser.add_argument("-wv", "--write_video", type=str, default="N",
                        help="write video to local file Y or N [Y or N]"
                        "(on by default)")
                           
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client() # Fixed Syntax
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL) # Syntax with parameter of ip port 
    return client

def check_input_type(input, id):
    """
    check input is video,image or cam
    """
    checkInputargs = input #string from args.input
    checkError = checkInputargs.find(".") #Verify If there is extension or other than CAM
    error_flag = False
    image_flag = False
    cap = None
    if checkInputargs == "CAM": # Check for cam
        cap = cv2.VideoCapture(id) # Assign CAM ID
        log.info("Performing inference on webcam video...")
    elif checkError is -1:  # Check for if there any  extension
        log.error("Error: invalid input or currupted file") # Error for no extension
        log.error("Use -h argument for help")
        error_flag = True
    else:
        path,ext= checkInputargs.rsplit(".",1) #find extension
        if ext == "bmp" or ext == "jpg": #supporeted ext.
            log.info("Performing inference on single image...")
            cap = cv2.VideoCapture(input)
            image_flag = True
        elif ext == "mp4" or ext == "MP4": #if not image feed video
            cap = cv2.VideoCapture(input) #Load local stream
            log.info("Performing inference on local video...")
        else:
            log.error("Image/Video formate not supported")
            error_flag = True
    return cap, error_flag, image_flag


def preprocess_frame(frame,height,width):
    p_frame = cv2.resize(frame, (height, width)) #Resize as per network input spec.
    p_frame = p_frame.transpose((2,0,1)) #swap channel cxhxw 
    p_frame = p_frame.reshape(1, *p_frame.shape) #add one axis 1 to make 4D shape for network input
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
    return color

def infer_on_stream(args, client):
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

    log.info("Selected Network input Layer type is "+str(type(net_input_shape))+" And shape is "+str(net_input_shape))
    log.info("Required input img size W "+str(net_input_shape[3])+" H "+str(net_input_shape[2]))

    # ### TODO: Handle the input stream ###
    # cap = cv2.VideoCapture(args.input)
    cap, error_flag, image_flag = check_input_type(args.input, args.cam_id) #call function
    if error_flag: # Check for invalid file extension
        log.error("Program stopped")
        return
    elif image_flag: #check for image 
        INPUT_IMAGE = args.input
        img = cv2.imread(INPUT_IMAGE)
        if (type(img) is not np.ndarray):  #check for if image read empty same as img.empty()
            log.error("Error: Invalid image or path")
            log.error("Use -h argument for help")
            return
    else:
        cap.open(args.input)

    # Get input feed height and width
    img_width = int(cap.get(3))
    img_height = int(cap.get(4))

    if img_width < 1 or img_width is None: # If input path is wrong
        log.error("Error! Can't read Input: Check path")
        return

    log.info("feed frame size W "+str(img_width) + " H " + str(img_height))

    # Initialize video writer if video mode
    if args.write_video is "Y": # only if args given Y
        if not image_flag:
            # Video writer Linux
            log.info("---Opencv video writer debug LIN---")
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (img_width,img_height))
            log.info("-------------------------------")

    # Initialized varible utilized inside loop
    frame_count = 0 
    total_people_count = 0
    last_state = 0
    delay_on = 0
    delay_off = (time.time() * 1000) # Initialize timer before loop to get actual time
    delay_diff_on = 0
    delay_diff_off = 0
    duration = 0
    duration_timebase = 0
    duration_fpsbase = 0
    count_people_image = 0

    # Second counting timer initialized
    sec_on = (time.time() * 1000) # Timer for update stat on terminal START
    sec_diff = 0
    cv_drawstate_time_s = 0
    cv_drawstate_time_e = 0
    count_flag = False

    frame_count_onstate = 0
    frame_count_offstate = 0

    # Accuracy Log
    log_acount = 0
    log_frame_no = []
    log_person_counted = []
    log_duration_fpsbase = []
    log_duration_timebase = []
    log_infer_time = []

    # error_log = {'MuliBoxDetected':{}}
    log_ecount = 0 #counter for error log in case of multiple box count
    log_multicounted = []

    # ### TODO: Loop until stream is over ###
    while cap.isOpened():
        frame_count += 1 # Global frame Count no of frame processed.
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(1)

        ### TODO: Read from the video capture ###
        ### TODO: Pre-process the image as needed ###
        p_frame = preprocess_frame(frame,net_input_shape[3],net_input_shape[2]) #from extracted input function
        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(p_frame)
        ### TODO: Wait for the result ###
        inferreq_start_time = (time.time() * 1000) # Timer for inference START
        if infer_network.wait() == 0:
            inferreq_end_time = (time.time() * 1000) - inferreq_start_time # Timer for inference END
            log_infer_time.append(float("{:.2f}".format(inferreq_end_time)))
            
            ### TODO: Get the results of the inference request ###
            blob, result = infer_network.get_output()

            # If model outputs multiple blob, print available blob infirmation
            if frame_count == 1: # Print only Once
                for name,output_ in blob.items(): #Find the possible BLOBS for name, 
                    log.info ("The name of available blob is :"+str(name))

            ### TODO: Extract any desired stats from the results ###
            color = selectBoxcolor(args.box_color)
            cv_drawboxtime_s = (time.time() * 1000) # Timer for drawing box on frame START
            frame, count_box, countmultipeople = draw_boxes(frame, result, img_width, img_height, color, args.prob_threshold)
            cv_drawboxtime_e = (time.time() * 1000) - cv_drawboxtime_s #Timer for drawing box on frame END
            
            count_people_image = countmultipeople # Variable For image stat only 
            ### TODO: Calculate and send relevant information on ###
            if count_box != last_state: #Anythinkg under this will executed only once if state changes.
                log_acount += 1 # increase stat change counter
                if count_box == 1:
                    count_flag = True # Flag for verify if counting 
                    delay_on = (time.time() * 1000)  # Timer for on delay START
                    delay_diff_off = (time.time() * 1000) - delay_off # Timer for off delay END
                    delay_diff_on = 0 # Timer for on delay RESET   

                    frame_count_onstate = frame_count # Frame count is Global FPS counter
                    frame_count_offstate = frame_count - frame_count_offstate # Calculates the difference
                else:
                    count_flag = False
                    delay_diff_on = (time.time() * 1000) - delay_on    # Timer for on delay END
                    delay_off = (time.time() * 1000)  # Timer for off delay START
                    delay_diff_off = 0 # Timer for off delay RESET

                    frame_count_onstate = frame_count - frame_count_onstate # Calculates the difference
                    frame_count_offstate = frame_count

                if delay_diff_on > args.delay_band:
                    total_people_count += 1 # Debug is placed above because count is not added yet.
                    duration_timebase = delay_diff_on / 1000 # Convert to Sec.
                    duration_fpsbase = frame_count_onstate / args.fps # Local use
                    duration = duration_fpsbase # global set 

                    # Accuracy log, individual list log, termianl friendly
                    log_person_counted.append(total_people_count)
                    log_duration_timebase.append("{:.2f}".format(duration_timebase))
                    log_duration_fpsbase.append(duration_fpsbase)
                    log_frame_no.append(frame_count) # Log frame no of video 

                    ### current_count, total_count and duration to the MQTT server ###
                    ### Topic "person": keys of "count" and "total" ###
                    client.publish("person", json.dumps({"total":total_people_count}))
                    ### Topic "person/duration": key of "duration" ###
                    client.publish("person/duration", json.dumps({"duration": duration}))
                client.publish("person", json.dumps({"count": countmultipeople}))

                last_state = count_box

            else:
                if countmultipeople not in (0,1): #In case of multiple people detected
                    log_ecount += 1 # Increase error counter
                    # Nested list Frame and multipeople people count
                    log_multicounted.append(['F: '+ str(frame_count) + ' C: ' + str(countmultipeople)])

        
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
            # log.info"time in ms: ",sec_diff) # Debug
            if sec_diff > 1000 or sec_diff > 2000: # update stat roughly every 1 sec.
                os.system('cls' if os.name == 'nt' else 'clear') # Clear the terminal
                print() # Blank print
                print("Video feed is OFF, Terminal will refresh every sec.")
                print("Press ctlr+c to stop execution.")
                print("Checkout log_xxx.txt for stats.")
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

                delay_on = delay_on + inferreq_end_time + cv_drawboxtime_e + cv_drawstate_time_e

        else:
                delay_off = delay_off + inferreq_end_time + cv_drawboxtime_e + cv_drawstate_time_e


        ### TODO: Send the frame to the FFMPEG server ###
        # Write video or image file
        if not image_flag:
            if args.toggle_video is "ON":
                sys.stdout.buffer.write(frame) # Send to ffmpeg
                sys.stdout.flush() # Send to ffmpeg
                # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                # cv2.imshow('frame',frame)
            if args.write_video is "Y":
                out.write(frame) 
        else:
            ### TODO: Write an output image if `single_image_mode` ###
            cv2.imwrite('output_image.jpg', frame)
            print("Image saved sucessfully!")

        if key_pressed == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()

    # Dump to log txt file
    log.info("Last frame prcessed no: "+ str(frame_count))
    log.info("-----AccuracyLog-----")
    if len(log_person_counted) > 1: # Only if counting single person 
        log.info("No Of person:")
        log.info(str(log_person_counted))
        # log.info("Duration stayed timebase:")
        # log.info(str(log_duration_timebase))
        log.info("Duration stayed fpsbase:")
        log.info(str(log_duration_fpsbase))
        log.info("Frame No.:")
        log.info(str(log_frame_no))
        log_infer_time = np.array(log_infer_time) # Convert list to np array
        log.info("Inference time:[min max avg.]")
        log.info(str([log_infer_time.min(),log_infer_time.max(),(float("{:.2f}".format(np.average(log_infer_time))))]))
    else:
        log.info("N/A")
        log_infer_time = np.array(log_infer_time) # Convert list to np array
        log.info("Inference time:[min max avg.]")
        log.info(([log_infer_time.min(),log_infer_time.max(),(float("{:.2f}".format(np.average(log_infer_time))))]))
        log.info("-----Error log-----")
        if len(log_multicounted) < 10 and len(log_multicounted) > 1: # Only if counting single person
            log.info("Frame No: Count")
            log.info(str(log_multicounted))
        else:
            log.info("N/A")
        log.info("-----Finish!------")



def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    # This is different method so do not use .m type attributes instead use whole name.
    args = build_argparser().parse_args()
    
    log.info("Commandline Arguments received")
    log.info("-----Information-----")
    log.info("Model path:"+ str(args.model))
    log.info("Video/Image path:"+ str(args.input))
    log.info("Video fps:"+str(args.fps))
    log.info("Device:"+str(args.device))
    log.info("CPU Ext. path:"+str(args.cpu_extension))
    log.info("BoundingBox color:"+str(args.box_color))
    log.info("Confidence:"+str(args.prob_threshold))
    log.info("Alarm People count:"+str(args.alarm_people))
    log.info("Alarm Person duration Sec.:"+str(args.alarm_duration))
    log.info("Web cam ID(If any):"+str(args.cam_id))
    log.info("Delay Band(ms):"+str(args.delay_band))
    log.info("Toggle video feed on/off:"+str(args.toggle_video))
    log.info("Write output to video file Y or N:"+str(args.write_video))
    log.info("-----------------------")
    
            
    # Connect to the MQTT server
    client = connect_mqtt()

    # Perform inference on the input stream
    infer_on_stream(args, client) 
    


if __name__ == '__main__':
    main()
