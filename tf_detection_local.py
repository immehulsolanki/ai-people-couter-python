# TensorFlow Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector
"""A.I People Counter TensorFlow Implementation."""
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
import cv2
import numpy as np
from argparse import ArgumentParser
import tensorflow as tf
tf.contrib.resampler
import tensorflow_core



BOXCOLOR = {'RED':(0,0,255),'GREEN':(0,255,0),'BLUE':(255,0,0),'WHITE':(255,255,255),'BLACK':(0,0,0)}

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

#======================== TF implementation  START ========================================

class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
    
    def processFrame(self, image):
            # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image, axis=0)
            # Actual detection.
            start_time = (time.time()*1000) #  inference time START
            (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_np_expanded})
            end_time = (time.time()*1000)  # inference time END

            # Debug Infere request time to terminal
            #print("Elapsed Time:"+ str("{:.2f}".format(end_time-start_time)) + "ms")
            
            inferreq_end_time = end_time - start_time

            im_height, im_width,_ = image.shape
            boxes_list = [None for i in range(boxes.shape[1])]
            for i in range(boxes.shape[1]):
                boxes_list[i] = (int(boxes[0,i,0] * im_height),
                            int(boxes[0,i,1] * im_width),
                            int(boxes[0,i,2] * im_height),
                            int(boxes[0,i,3] * im_width))

            return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0]), inferreq_end_time

    def close(self):
        self.sess.close()
        self.default_graph.close()

#======================== TF implementation  END ==============================================


def check_input_type(input, id):
    """
    check input is video,image or cam
    """
    
    checkInputargs = input 
    checkError = checkInputargs.find(".") 
    error_flag = False
    image_flag = False
    cap = None
    if checkInputargs == "CAM": 
        cap = cv2.VideoCapture(id) 
        print("Performing inference on webcam video...")
    elif checkError is -1:  
        print("Error: invalid input or currupted file")
        print("Use -h argument for help")
        error_flag = True
    else:
        path,ext= checkInputargs.rsplit(".",1) 
        if ext == "bmp" or ext == "jpg": 
            print("Performing inference on single image...")
            cap = cv2.VideoCapture(input)
            image_flag = True
        elif ext == "mp4" or ext == "MP4":
            cap = cv2.VideoCapture(input) 
            print("Performing inference on local video...")
        else:
            print("Image/Video formate not supported")
            error_flag = True
    return cap, error_flag, image_flag


def draw_framelinegreen(frame,height,width): 
    """
    Draw normal Green frame on video
    """
    # Draw line top left and right
    cv2.line(frame, (0, 0), (0,int(height/10)), (0,255,0),10)
    cv2.line(frame, (0, 0), (int(height/10),0), (0,255,0),10)
    cv2.line(frame, (width, 0), (width-int(height/10),0), (0,255,0),10)
    cv2.line(frame, (width, 0), (width,int(height/10)), (0,255,0),10)

    # Draw line bottom left and right
    cv2.line(frame, (0, height), (0,height-int(height/10)), (0,255,0),10)
    cv2.line(frame, (0, height), (int(height/10),height), (0,255,0),10)
    cv2.line(frame, (width, height), (width-int(height/10),height), (0,255,0),10)
    cv2.line(frame, (width, height), (width,height-int(height/10)), (0,255,0),10)
    return frame

def draw_framelinered(frame,height,width): #Better to pass Color parameter
    """
    Draw alert red frame on video
    """
    # Draw line top left and right
    cv2.line(frame, (0, 0), (0,int(height/10)), (0,0,255),10)
    cv2.line(frame, (0, 0), (int(height/10),0), (0,0,255),10)
    cv2.line(frame, (width, 0), (width-int(height/10),0), (0,0,255),10)
    cv2.line(frame, (width, 0), (width,int(height/10)), (0,0,255),10)

    # Draw line bottom left and right
    cv2.line(frame, (0, height), (0,height-int(height/10)), (0,0,255),10)
    cv2.line(frame, (0, height), (int(height/10),height), (0,0,255),10)
    cv2.line(frame, (width, height), (width-int(height/10),height), (0,0,255),10)
    cv2.line(frame, (width, height), (width,height-int(height/10)), (0,0,255),10)
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


def infer_on_stream(args):
    """
    Performance test code for tf detection models
    """
    # Set Probability threshold for detections
    model_path = args.input
    odapi = DetectorAPI(path_to_ckpt=args.model)
    threshold = args.prob_threshold

    cap, error_flag, image_flag = check_input_type(args.input, args.cam_id) 
    if error_flag: 
        print("Program stopped")
        return
    elif image_flag: 
        INPUT_IMAGE = args.input
        img = cv2.imread(INPUT_IMAGE)
        if (type(img) is not np.ndarray):  
            print("Error: Invalid image or path")
            print("Use -h argument for help")
            return
    else:
        cap.open(args.input)

    # Get input feed height and width
    img_width = int(cap.get(3))
    img_height = int(cap.get(4))

    if img_width < 1 or img_width is None: 
        print("Error! Can't read Input: Check path")
        return

    print("feed frame size W",img_width,"H",img_height)

    # Initialize video writer if video mode
    if not image_flag:
        # Video writer Windows10
        print("---Opencv video writer debug WIN---")
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter('out.mp4', fourcc, args.fps, (img_width,img_height))
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
    delay_off = (time.time() * 1000) 
    delay_diff_on = 0
    delay_diff_off = 0
    duration = 0
    duration_timebase = 0
    duration_fpsbase = 0
    count_people_image = 0

    # Second counting timer initialized
    sec_on = (time.time() * 1000) 
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

    # error_log 
    log_ecount = 0 
    log_multicounted = []

    # Duration manual count [13, 21, 18, 11, 27, 12]

    # ### TODO: Loop until stream is over ###
    while cap.isOpened():
        frame_count += 1
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(1)

#============================== TF implementation  START ===================================
        # TF Preprocess frame, Explicit resize not require
        boxes, scores, classes, num, inferreq_end_time  = odapi.processFrame(frame)

        color = selectBoxcolor(args.box_color)
        cv_drawboxtime_s = (time.time() * 1000) # Timer for drawing box on frame START

        count_box = 0
        countmultipeople = 0
        for i in range(len(boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:
                count_box = 1
                countmultipeople += 1
                box = boxes[i]
            
                # Scaling already applied in TF proccesing code.
                xmin = box[1]
                ymin = box[0]
                xmax = box[3]
                ymax = box[2]

                label = "Person"+str(countmultipeople)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 1) 
                cv2.rectangle(frame, (xmin, ymin), (xmin+90, ymin+10), color, -1) 
                cv2.putText(frame, label, (xmin,ymin+10),cv2.FONT_HERSHEY_PLAIN, 0.8, BOXCOLOR['BLACK'], 1) 

        cv_drawboxtime_e = (time.time() * 1000) - cv_drawboxtime_s 
#================================= TF implementation  END ====================================

        log_infer_time.append(float("{:.2f}".format(inferreq_end_time)))
        
        count_people_image = countmultipeople 
        if count_box != last_state: 
            log_acount += 1 
            if count_box == 1:
                count_flag = True 
                delay_on = (time.time() * 1000)  
                delay_diff_off = (time.time() * 1000) - delay_off
                delay_diff_on = 0 
                frame_count_onstate = frame_count 
                frame_count_offstate = frame_count - frame_count_offstate 
            else:
                count_flag = False
                delay_diff_on = (time.time() * 1000) - delay_on    
                delay_off = (time.time() * 1000) 
                delay_diff_off = 0 

                frame_count_onstate = frame_count - frame_count_onstate 
                frame_count_offstate = frame_count

            # For Debug if state changes then only update values
            # print("update on",delay_diff_on) 
            # print("update off",delay_diff_off) 
            # print(['frame_count_onstate: '+ str(frame_count_onstate), 'frame_count_offstate: '+ str(frame_count_offstate)])

            if delay_diff_on > args.delay_band:
                total_people_count += 1 # Debug is placed above because count is not added yet.
                duration_timebase = delay_diff_on / 1000 
                duration_fpsbase = frame_count_onstate / args.fps 
                duration = duration_fpsbase 

                # Debug Delay difference Update only when counting ++
                # print("count++ "+ " DDON: " + str("{:.2f}".format(delay_diff_on)) + " DDOF: " + str("{:.2f}".format(delay_diff_off)), 
                #     "duration: " + str("{:.2f}".format(duration)) + "Sec.") # Debug When count++
                # Debug Count status Update only when counting ++
                # print(['FrameNo:'+str(frame_count),'CurrentCount: '+
                #     str(countmultipeople),'TotalCount: '+str(total_people_count),'duration_timebase: '+str("{:.2f}".format(duration_timebase))])
                # print('duration_fpsbase: '+ str(frame_count_onstate / args.fps))

                # Accuracy log, individual list log, termianl friendly
                log_person_counted.append(total_people_count)
                log_duration_timebase.append("{:.2f}".format(duration_timebase))
                log_duration_fpsbase.append(duration_fpsbase)
                log_frame_no.append(frame_count) # Log frame no of video 

            last_state = count_box

            # state log for all variable changes when stat changes
            # Debug if state changes 1 or 0 everytime, delay diff On/Off changes 
            # print(['Instate: '+ str(count_box),'delaydifOn: '+ str("{:.2f}".format(delay_diff_on)),
            #     'delaydifOff: '+ str("{:.2f}".format(delay_diff_off))])
            # print(['FrameNo:'+str(frame_count),'CurrentCount: '+
            #     str(countmultipeople),'TotalCount: '+str(total_people_count),'duration: '+str("{:.2f}".format(duration))])
            # print() # Add blank print for space
        else:
            if countmultipeople not in (0,1): 
                log_ecount += 1 
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
                label2 = "Average Time stayed: " + str("{:.2f}".format(duration)) + "Sec." 
            cv2.putText(frame, label2, (15,40),cv2.FONT_HERSHEY_COMPLEX, 0.4, BOXCOLOR['BLUE'], 1)

            # People count exceed alarm
            if countmultipeople > args.alarm_people or duration > args.alarm_duration:
                draw_framelinered(frame,img_height,img_width)
                if countmultipeople > args.alarm_people:
                    label3 = "Alarm: people count limit exceeded! limit: "+ str(args.alarm_people)
                    cv2.putText(frame, label3, (15,50),cv2.FONT_HERSHEY_COMPLEX, 0.4, BOXCOLOR['RED'], 1)
                else:
                    label4 = "Alarm: Person stayed longer! limit: " + str(args.alarm_duration) + "Sec." 
                    cv2.putText(frame, label4, (15,60),cv2.FONT_HERSHEY_COMPLEX, 0.4, BOXCOLOR['RED'], 1)
            else:
                draw_framelinegreen(frame,img_height,img_width)
                # Draw cv process time
            label5 = "CV Frame process time: " + str("{:.2f}".format(cv_drawboxtime_e + cv_drawstate_time_e)) + "ms" 
            cv2.putText(frame, label5, (15,70),cv2.FONT_HERSHEY_COMPLEX, 0.4, BOXCOLOR['BLUE'], 1)
            cv_drawstate_time_e = (time.time() * 1000) - cv_drawstate_time_s 
        else:
             # Stats of time of cv processing on image frame
            sec_diff = (time.time() * 1000) - sec_on  
            # print("time in ms: ",sec_diff) # Debug
            if sec_diff > 1000 or sec_diff > 2000: 
                os.system('cls' if os.name == 'nt' else 'clear')
                print() 
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
                        print("Person stayed longer! limit: " + str(args.alarm_duration) + "Sec.")
                        print("##################") 
                print("-----Stats for time -----") 
                print("Inference Time(ms):","{:.2f}".format(inferreq_end_time))
                print("Draw boundingBox time(ms):", "{:.2f}".format(cv_drawboxtime_e))
                print("Draw state time(ms):", "{:.2f}".format(cv_drawstate_time_e))
                print("--------------------------") 
                sec_on = (time.time() * 1000) 
                sec_diff = 0 

        # Adjusting timers with inference and cv processing time to fix counting and duration.
        if count_flag:
                
                delay_on = delay_on + inferreq_end_time + cv_drawboxtime_e + cv_drawstate_time_e
                
        else:
                delay_off = delay_off + inferreq_end_time + cv_drawboxtime_e + cv_drawstate_time_e

        # Write video or image file
        if not image_flag:
            if args.toggle_video is "ON":
                cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                cv2.imshow('frame',frame)
            #out.write(frame)  #Enable this to write video
        else:
            cv2.imwrite('output_image.jpg', frame)
            print("Image saved sucessfully!")

        if args.toggle_video is "ON":
            a = None

        if key_pressed == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("Last frame prcessed no: ",frame_count)
    print("-----AccuracyLog-----")
    if len(log_person_counted) >= 1: 
        print("No Of person:")
        print(log_person_counted)
        # print("Duration stayed timebase:")
        # print(log_duration_timebase)
        print("Duration stayed fpsbase:")
        print(log_duration_fpsbase)
        print("Frame No.:")
        print(log_frame_no)
        log_infer_time = np.array(log_infer_time)
        print("Inference time:[min max avg.]")
        print([log_infer_time.min(),log_infer_time.max(),(float("{:.2f}".format(np.average(log_infer_time))))])
    else:
        print("N/A")
        log_infer_time = np.array(log_infer_time)
        print("Inference time:[min max avg.]")
        print([log_infer_time.min(),log_infer_time.max(),(float("{:.2f}".format(np.average(log_infer_time))))])

    print("-----Error log-----")
    if len(log_multicounted) < 10 and len(log_multicounted) > 1: 
        print("Frame No: Count")
        print(log_multicounted)
    else:
        print("N/A")
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
    print("Video fps:",args.fps)
    print("BoundingBox color:",args.box_color)
    print("Confidence:",args.prob_threshold)
    print("Alarm People count:",args.alarm_people)
    print("Alarm Person duration Sec.:",args.alarm_duration)
    print("Web cam ID(If any):",args.cam_id)
    print("Delay Band(ms):", args.delay_band)
    print("Toggle video feed on/off:",args.toggle_video)
    print("-----------------------")

    # Perform inference on the input stream
    infer_on_stream(args)



if __name__ == '__main__':
    main()
