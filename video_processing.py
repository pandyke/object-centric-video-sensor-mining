import uuid
import pandas as pd
import numpy as np
import time
import datetime
import math
import cv2
import torch
from ultralytics import YOLO
import os
import glob

def init_object_tracking(filepath):
    model = load_model("yolov8n")
    tracker = 'botsort.yaml' #available trackers in YOLO: 'bytetrack.yaml', 'botsort.yaml'
    device = 'cuda' if torch.cuda.is_available() else 'cpu' #torch.cuda.is_available() is True if GPU supports cuda
    stream = cv2.VideoCapture(filepath)
    return stream, model, tracker, device

def load_model(model_name):
    """
    load_model loads one of the predefined object recognition models

    :model_name: The name of one of the predefined models as string, e.g., 'yolov5s'
    :return: the loaded model
    """
    if model_name == "yolov5s":
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    elif model_name == "yolov8n":
        model = YOLO('yolov8n.pt')
    else:
        #define a standard model here
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    return model

def selectTracker(tracker_name):
    """
    selectTracker loads one of the predefined object trackers

    :tracker_name: The name of one of the predefined trackers as string, e.g. one of ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    :return: the created tracker
    """

    if tracker_name == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    elif tracker_name == 'MIL':
        tracker = cv2.TrackerMIL_create()
    elif tracker_name == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif tracker_name == 'TLD':
        tracker = cv2.TrackerTLD_create()
    elif tracker_name == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    elif tracker_name == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    elif tracker_name == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    elif tracker_name == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    else:
        #define a standard tracker here
        tracker = cv2.TrackerKCF_create()
    
    return tracker

def object_tracking(stream, model, tracker, device, confidence=0.3):
    """
    object_tracking performs object detection and object tracking on the video stream and returns relevant information in a dataframe,
        e.g., the detected objects per frame together with attributes.

    :stream: The cv2 video stream
    :model: The model that is used for object detection
    :tracker: The tracking method that is used for object tracking
    :device: The device that the model uses, e.g., GPU
    :confidence: The confidence threshhold that is used for object detection/tracking
    :return: a dataframe where each row is identified by a combination of the frame number and a detected object in the video stream, and for each recognized object,
        the object ID, recognized object class, confidence, and bounding box information (normalized on frame/image size).
        Extra columns for further manual labeling of object ID and object class are already added
    """
    
    amount_of_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT)) #get total number of frames
    start_time = time.time()
    model.to(device)
    tracking_list = []

    fps = stream.get(cv2.CAP_PROP_FPS)

    while stream.isOpened(): # Run until stream is out of frames
        success, frame = stream.read()
        curr_frame_number = stream.get(cv2.CAP_PROP_POS_FRAMES) #retrieves the current frame number
        if success:
            results = model.track(frame, persist=True, tracker=tracker, conf=confidence, iou=0.5) #run tracking on the frame, persisting tracks between frames

            #get relevant attributes per frame
            frame_result = results[0]
            #print(curr_frame_number, frame_result.boxes.id, frame_result.boxes.cls, frame_result.boxes.conf, frame_result.boxes.xywhn)
            #note: x and y mark the center of the bounding box and w and h are width and height. n indicates that the values are normalized on the original frame size
            tracking_list.append([curr_frame_number, frame_result.boxes.id, frame_result.boxes.cls, frame_result.boxes.conf, frame_result.boxes.xywhn])

        else:
            break
    
    #create and format final dataframe containing all frames and relevant attributes
    df_raw_results = pd.DataFrame(tracking_list, columns = ['frame_number', 'object_ids', 'object_classes', 'object_confidence', 'bounding_box_coords'])
    df_raw_results["frame_number"] = df_raw_results["frame_number"].astype(int) #convert frame number to integer
    df_raw_results["object_ids"] = df_raw_results["object_ids"].map(lambda x: x.tolist()) #convert object IDs to list
    df_raw_results["object_classes"] = df_raw_results["object_classes"].map(lambda x: x.tolist()) #convert object classes to list
    df_raw_results["object_confidence"] = df_raw_results["object_confidence"].map(lambda x: x.tolist()) #convert object confidence to list
    df_raw_results["bounding_box_coords"] = df_raw_results["bounding_box_coords"].map(lambda x: x.tolist()) #convert bounding box coordinates to list

    #convert the df that one row is identified by a combination of frame number and detected object (i.e., one object per row)
    frame_object_list = []
    for index, row in df_raw_results.iterrows():
        curr_frame_num = row['frame_number']
        for idx, obj_id in enumerate(row['object_ids']):
            curr_obj_class = row['object_classes'][idx]
            curr_obj_conf = row['object_confidence'][idx]
            curr_bb_ccords = row['bounding_box_coords'][idx]
            frame_object_list.append([curr_frame_num, obj_id, curr_obj_class, curr_obj_conf, curr_bb_ccords])
    df_tracking_results = pd.DataFrame(frame_object_list, columns = ['frame_number', 'object_id', 'object_class', 'object_confidence', 'bounding_box'])
    df_tracking_results['object_id'] = df_tracking_results['object_id'].astype(int)
    df_tracking_results['object_class'] = df_tracking_results['object_class'].astype(int)
    class_names_dict = model.names #the dictionary containing the class numbers and corresponding names
    df_tracking_results["object_class"] = df_tracking_results["object_class"].map(lambda x: class_names_dict.get(x))

    #Add columns for manual assignment of object ID and object class that can be changed later
    df_tracking_results["object_id_manual"] = "" #df_tracking_results["object_id"]
    df_tracking_results["object_class_manual"] = "" #df_tracking_results["object_class"]

    #Add column for object attributes for later
    df_tracking_results["object_attr_manual"] = df_tracking_results.apply(lambda x: [], axis=1)
    df_tracking_results["ignore_object"] = False

    #release everything
    #stream.release()
    cv2.destroyAllWindows()

    end_time = time.time()
    print("Total amount of frames:", amount_of_frames)
    print("Fps of video:", fps)
    vid_length_seconds = amount_of_frames/fps
    print("Video length in seconds:", np.round(vid_length_seconds,2))
    print("Total duration (H:M:S):", time.strftime('%H:%M:%S', time.gmtime(end_time-start_time)))
    print("Fps of tracking:", np.round(amount_of_frames/(end_time-start_time), 2))
    print("Duration factor (Tracking duration / video length):", np.round((end_time-start_time)/vid_length_seconds,2))

    return df_tracking_results

def annotate_image(stream, frame_number, obj_id, obj_class, obj_conf, bounding_box):
    """
    annotate_image takes a stream, a frame number and relevant infos on a detected object of interest and returns a custom annotated image of that object in that frame

    :stream: The cv2 video stream
    :frame_number: The frame number in which the object is detected and which should be annotated
    :obj_id: The id of the object of interest, assigned to it at object tracking
    :obj_class: The object class of the object of interest, assigned to it at object tracking
    :obj_conf: The confidence of object detection on that object
    :bounding_box: The bounding box of the object of interest, found at object tracking
    :return: The custom annotated image of the object of interest in the defined frame
    """

    #amount_of_frames = stream.get(cv2.CAP_PROP_FRAME_COUNT) #get total number of frames
    #get specific frame from frame number
    stream.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
    success, frame = stream.read()
    if success:
        frame_height, frame_width = frame.shape[0], frame.shape[1]
        #note that the bounding boxes are given as xywhn, where x and y mark the center of the bounding box and w and h are width and height.
        #n indicates that the values are normalized on the original frame size
        width = int(bounding_box[2]*frame_width)
        height = int(bounding_box[3]*frame_height)
        x_top_left = int(bounding_box[0]*frame_width-(width/2))
        y_top_left = int(bounding_box[1]*frame_height-(height/2))

        cv2.rectangle(frame,(x_top_left,y_top_left),(x_top_left+width,y_top_left+height),(0,255,0),2) #top left corner, bottom right corner, color, line thickness
        #Decide if text should be placed above bounding box (standard) or below the top line/inside bounding box (if bounding box is at the top of the frame)
        if y_top_left > 20:
            y_coord_text = y_top_left-4
            rect_start = (x_top_left,y_top_left-20)
            rect_end = (x_top_left+width, y_top_left)
        else:
            y_coord_text = y_top_left+20
            rect_start = (x_top_left,y_top_left)
            rect_end = (x_top_left+width, y_top_left+25)
        cv2.rectangle(frame, rect_start, rect_end, (0,255,0), -1)
        cv2.putText(frame,"Id:"+str(obj_id)+", "+obj_class+", "+str(round(obj_conf,2)),
                    (x_top_left,y_coord_text),cv2.FONT_HERSHEY_DUPLEX,0.7,(0,0,0),1) #coords, font, fontScale, color, thickness
        #cv2.imshow("Image for labeling", frame) #for debugging
        #cv2.waitKey(0) #for debugging
    else:
        print("No success in stream.read()")
    #stream.release()
    cv2.destroyAllWindows()
    return frame

def create_collage(images):
    """
    create_collage takes a list of 1, 2, 4, or 9 images and creates a collage of them returning one image

    :images: A list of images. Must include 1, 2, 4, or 9 images
    :return: An image which is a collage of the images given as inputs
    """

    number_images = len(images)
    if number_images == 0:
        print("No image given in list. Aborting function")
        return
    elif number_images == 1:
        collage = images[0]
    elif number_images == 2:
        collage = cv2.hconcat(images)
    elif number_images == 4:
        im_1, im_2, im_3, im_4 = images
        collage = cv2.vconcat([cv2.hconcat([im_1, im_2]), cv2.hconcat([im_3, im_4])])
    elif number_images == 9:
        im_1, im_2, im_3, im_4, im_5, im_6, im_7, im_8, im_9 = images
        collage = cv2.vconcat([cv2.hconcat([im_1, im_2, im_3]), cv2.hconcat([im_4, im_5, im_6]), cv2.hconcat([im_7, im_8, im_9])])
    else:
        collage = images[0] #only take first image

    return collage

def object_labeling_image_prep(stream, tracking_results_df, session_path, video_session_id, sample_size):
    """
    object_labeling_image_prep takes each recognized object in the stream by object ID and prepares and saves an example collage of randomly selected pictures of the object 
        with a bounding box around the object and further infos.

    :stream: The cv2 video stream
    :tracking_results_df: The dataframe from the object tracking step, containing all detected objects in all frames, together with attributes
    :session_path: The path to the current session folder including a '/' at the end.
    :video_session_id: The name of the current video session folder
        Note: In the  folder the subfolder 'labeling_images' has to exist for this function to work
        Note: Image files with the same name in that subfolder will be overwritten
    :sample_size: Amount of frames per object that are displayed in the collage. Must be 1, 2, 4, 9
    """
    if sample_size not in {1, 2, 4, 9}:
        print("Error: Sample size must be 1, 2, 4, or 9. Function aborted")
        return

    image_saving_path = session_path + video_session_id + "/labeling_images/"

    object_ids = tracking_results_df['object_id'].unique()
    #print(f"object_labeling_image_prep: Number of object IDs {len(object_ids)}")
    for curr_obj_id in object_ids: #iterate over every uniquely identified object
        all_rows_curr_object = tracking_results_df.loc[tracking_results_df['object_id'] == curr_obj_id]
        if len(all_rows_curr_object) >= sample_size:
            curr_sample = all_rows_curr_object[['frame_number', 'object_class', 'object_confidence', 'bounding_box']].sample(
                n=sample_size, random_state=42) #get a random sample of frames with current object in it
        else:
            sample_size = 1
            curr_sample = all_rows_curr_object[['frame_number', 'object_class', 'object_confidence', 'bounding_box']].sample(
                n=sample_size, random_state=42) #get a random sample of frames with current object in it
            print(f"object_labeling_image_prep: Sample size set to 1 because there are not enough frames with object in it. {len(all_rows_curr_object)} frames with object.")
        all_ann_images = []
        for curr_n in range(sample_size): #for every sample frame get the annotated image
            curr_frame_number, curr_obj_class, curr_obj_conf, curr_bb = curr_sample.iloc[curr_n]
            annotated_image = annotate_image(stream, curr_frame_number, curr_obj_id, curr_obj_class, curr_obj_conf, curr_bb)
            all_ann_images.append(annotated_image)
        #make collage
        im_collage = create_collage(all_ann_images)
        #cv2.imwrite(image_saving_path + "ID" + str(curr_obj_id) + "_" + curr_obj_class + '.jpg', im_collage) #save the collage image for labeling
        cv2.imwrite(image_saving_path + str(curr_obj_id) + '.jpg', im_collage) #save the collage image for labeling
        print("object_labeling_image_prep: collage for object created and saved to current session")
        
def object_labeling_post_annotator(tracking_results_df_path, old_obj_id, new_obj_id, new_obj_type, new_attr_list):
    """
    object_labeling_post_annotator takes the path to the old tracking results dataframe, and the old object ID
    (that was annotated in object tracking), as well as a new manually assigned object ID, object class and object attributes.
    The function labels the object of interest in the whole tracking results dataframe accordingly and saves it

    :tracking_results_df_path: The path to the current tracking_results.
    :old_obj_id: (Int) The old object ID (that was annotated in object tracking), e.g. 1
    :new_obj_id: (String) The new object ID
    :new_obj_type: (String) The new object Type
    :new_attr_list: (List) The new object Attributes as list
    """
    if new_obj_id == "":
        return
    tracking_results = pd.read_pickle(tracking_results_df_path) #load results of current session
    #print(tracking_results)
    #print(old_obj_id, new_obj_id, new_obj_type, new_attr_list)
    tracking_results['object_id_manual'] = tracking_results['object_id_manual'].astype(str)
    tracking_results.loc[tracking_results.object_id == old_obj_id, ['object_id_manual', 'object_class_manual', 'object_attr_manual']] = new_obj_id, new_obj_type, str(new_attr_list)
    tracking_results.to_pickle(tracking_results_df_path) #save and overwrite updated results in folder of current session

def manual_object_definer_prep(stream, session_path, video_session_id):
    """
    manual_object_definer_prep takes a frame in the middle of the stream and saves it as 'object_definition_sample_img.jpg' in the current session folder.
        This image can then exemplarily be used later to define manual objects at specific coordinates in the frame

    :stream: The cv2 video stream
    :session_path: The path to the current session folder including a '/' at the end.
    :video_session_id: The name of the current video session folder
        Note: Image files with the same name in that folder will be overwritten
    """
    amount_of_frames = stream.get(cv2.CAP_PROP_FRAME_COUNT) #get total number of frames
    frame_number = amount_of_frames//2 #take middle frame and round down
    stream.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1) #get specific frame from random frame number
    success, frame = stream.read()
    if success:
        cv2.imwrite(session_path + video_session_id + '/object_definition_sample_img.jpg', frame) #save the image to the session folder
        print("manual_object_definer_prep: example image for manual objects definition saved to current session")
    else:
        print("manual_object_definer_prep: No success in stream.read()")

def select_bounding_boxes(session_path, video_session_id):
    """
    select_bounding_boxes opens the sample image of the current video session and lets the user draw a rectangle for defining an own object.

    :session_path: The path to the current session folder including a '/' at the end.
    video_session_id: The name of the current video session folder
    :return: Bounding box coordinates in two formats:
        1. xywh, where x and y mark the top left corner of the bounding box and w and h are width and height.
        2. xywh, where x and y mark the center of the bounding box and w and h are width and height. The values are normalized on the original frame/image size.
    """
    sample_img_path = session_path + video_session_id + '/object_definition_sample_img.jpg'
    image = cv2.imread(sample_img_path)
    img_width, img_height = image.shape[1], image.shape[0]
    #cv2.WINDOW_FULLSCREEN or cv2.WINDOW_AUTOSIZE or cv2.WINDOW_NORMAL in combination with resizeWindow
    cv2.namedWindow("Draw rectangle for bounding box of new object and press ENTER or SPACE twice. Press 'C' twice to cancel.", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Draw rectangle for bounding box of new object and press ENTER or SPACE twice. Press 'C' twice to cancel.", 576,648)
    #SelectROi returns bounding boxes in this format: [10, 8, 1335, 104].
    #Format is xywh, where x and y mark the top left corner of the bounding box and w and h are width and height.
    bounding_boxes = cv2.selectROI("Draw rectangle for bounding box of new object and press ENTER or SPACE twice. Press 'C' twice to cancel.", image) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    bounding_boxes = list(bounding_boxes)
    bb_x_min, bb_y_min, bb_width, bb_height =  bounding_boxes[0], bounding_boxes[1], bounding_boxes[2], bounding_boxes[3], 
    bb_center_x = bb_x_min + (bb_width/2)
    bb_center_y = bb_y_min + (bb_height/2)
    bb_center_x_norm = bb_center_x/img_width
    bb_center_y_norm = bb_center_y/img_height
    bb_width_norm = bb_width/img_width
    bb_height_norm = bb_height/img_height
    bb_yolo_format = [bb_center_x_norm, bb_center_y_norm, bb_width_norm, bb_height_norm]

    return bounding_boxes, bb_yolo_format

def create_image_with_bounding_box(session_path, video_session_id, bb_coordinates):
    """
    create_image_with_bounding_box takes bounding box coordinates and draws it on the sample image of the current video session.

    :session_path: The path to the current session folder including a '/' at the end.
    video_session_id: The name of the current video session folder
    :bb_coordinates: Coordinates of the bounding box in the format [10, 8, 1335, 104]. Format is xywh, where x and y mark the top left corner of the bounding box
        and w and h are width and height.
    :return: Returns the coordinates of the bounding box for later use in the format: left, top, right, bottom (or: x_min, y_min, x_max, y_max)
    """
    sample_img = cv2.imread(session_path + video_session_id + '/object_definition_sample_img.jpg')
    bb_left, bb_top = bb_coordinates[0], bb_coordinates[1]
    bb_width, bb_height = bb_coordinates[2], bb_coordinates[3]
    bb_right = bb_left + bb_width
    bb_bottom = bb_top + bb_height
    sample_img = cv2.rectangle(img=sample_img, pt1=(bb_left, bb_top), pt2=(bb_right, bb_bottom),
                               color=(0,255,0), thickness=2)
    cv2.imwrite(session_path + video_session_id + '/curr_object_area.jpg', sample_img)
    return bb_left, bb_top, bb_right, bb_bottom

def manual_object_definer(tracking_results_df_path, defined_object):
    """
    manual_object_definer takes the path to the tracking results dataframe, and info on the manually defined object,
        including the manually assigned object ID, object class, bounding box coordinates, and object attributes.
        The function adds the manually defined object to every frame by inserting a row with the new object for every frame in the dataframe and saves the updated dataframe.

    :tracking_results_df_path: The path to the current session folder including a '/' at the end.
        Note: The results dataframe in this session folder has to be named 'tracking_results.pkl'
    defined_object: Info on the manually defined object, including the manually assigned object ID, object class, bounding box coordinates, and object attributes.
        Example: defined_object = ['object ID 1', 'object class 1', [0.4086, 0.7910, 0.5589, 0.3621], [{"name":"attrName", "value":"attrValue", "time":""}]]
        Note: The bounding box coordinates are of the form xywh, where x and y mark the center of the bounding box and w and h are width and height.
        The values are normalized on the original frame/image size.
    """

    tracking_results = pd.read_pickle(tracking_results_df_path + "tracking_results.pkl") #load results of current session
    tracking_results['object_id_manual'] = tracking_results['object_id_manual'].astype(str)
    amount_of_frames = tracking_results['frame_number'].max() #get number of frames
    data = {'bounding_box': [defined_object[2]]*amount_of_frames, 'object_id_manual': [defined_object[0]]*amount_of_frames,
            'object_class_manual': [defined_object[1]]*amount_of_frames, 'frame_number': range(1, amount_of_frames+1), 'object_attr_manual': [defined_object[3]]*amount_of_frames,
            'ignore_object':[False]*amount_of_frames}
    append_df = pd.DataFrame(data)
    tracking_results = pd.concat([tracking_results, append_df], ignore_index=True)
    tracking_results['object_id'] = tracking_results['object_id'].astype('Int64')
    tracking_results = tracking_results.sort_values(['frame_number', 'object_id'], ascending=[True, True], ignore_index=True) #sort final df by frame_number and old object_id
    tracking_results.to_pickle(tracking_results_df_path + "tracking_results.pkl") #save and overwrite updated df in folder of current session

def get_frame_timestamp(video_start_timestamp_str, video_fps, frame_number):
        """
        get_frame_timestamp takes the start timestamp of the video and the frames per second of the video and calculates the resulting timestamp for a specific frame
            by adding the elapsed milliseconds to the start timestamp

        :video_start_timestamp_str: The start timestamp of the video, i.e. in frame 0. Format as string: '1970-01-01T00:00:00.000000'
        :video_fps: The frames per second of the video
        :frame_number: The number of the frame that the timestamp should be calculated for
        :return: Returns the calculated timestamp as string for the frame of interest. Format: '1970-01-01T00:00:00.000000'
        """
        frame_ms_from_start = 1000.0*(frame_number/video_fps)
        video_start_timestamp = datetime.datetime.strptime(video_start_timestamp_str, '%Y-%m-%dT%H:%M:%S.%f')
        frame_timestamp = video_start_timestamp + datetime.timedelta(milliseconds=frame_ms_from_start)
        frame_timestamp_str = frame_timestamp.strftime('%Y-%m-%dT%H:%M:%S.%f')
        return frame_timestamp_str

def bounding_boxes_overlapping(bb_obj_1,bb_obj_2):
    """
    bounding_boxes_overlapping checks for the bounding boxes of two objects if they are overlapping or not and return True or False respectively.

    :bb_obj_1: The bounding box coordinates of object 1 as a list containing the four values xywh, where x and y mark the center of the bounding box and w and h are width and height.
    :bb_obj_2: The bounding box coordinates of object 2 as a list containing the four values xywh, where x and y mark the center of the bounding box and w and h are width and height.
    :return: True or False if bounding boxes are overlapping or not
    """

    mid_x_1, mid_y_1, width_1, height_1 = bb_obj_1[0], bb_obj_1[1], bb_obj_1[2], bb_obj_1[3]
    mid_x_2, mid_y_2, width_2, height_2 = bb_obj_2[0], bb_obj_2[1], bb_obj_2[2], bb_obj_2[3]

    most_left_1, most_right_1, most_up_1, most_down_1 = mid_x_1-(width_1/2), mid_x_1+(width_1/2), mid_y_1+(height_1/2), mid_y_1-(height_1/2)
    most_left_2, most_right_2, most_up_2, most_down_2 = mid_x_2-(width_2/2), mid_x_2+(width_2/2), mid_y_2+(height_2/2), mid_y_2-(height_2/2)

    if most_left_2 > most_right_1 or most_right_2 < most_left_1: #not overlapping
        return False
    elif most_down_2 > most_up_1 or most_up_2 < most_down_1: #not overlapping
        return False
    else:
        return True

def video_event_trigger_algorithm_standard(session_path, video_session_id, video_start_timestamp_str, video_fps, rules):
    """
    video_event_trigger_algorithm_standard is a standard algorithm that defines events based on the bounding boxes of tracked objects.
        The algorithm defines an event when the bounding boxes of two detected/tracked objects start to overlap and another event when the bounding boxes stop overlapping.
        A dataframe containing all events is returned.
        The algorithm serves as an example and can be expanded for individual needs and specific use cases in the 'video_event_trigger_algorithm_custom' function.

    :session_path: The path to the current session folder including a '/' at the end.
    :video_session_id: The name of the current video session folder. Note: The results dataframe in this folder has to be named 'tracking_results.pkl'
    :video_start_timestamp_str: The exact timestamp when the video started, i.e., the timestamp of frame 0. Format as string: '1980-01-01T10:28:00.000000'
    :video_fps: The frames per second of the video
    :rules: A list of lists containing the rules for event creation, i.e., a combination of two object IDs and the corresponding activity name.
        E.g., [[objectID1, objectID2, activity name], [person_1, laptop_5, Person interacts with laptop]]
    :return: Returns a list of new events with every event in this format:
        {"id": "event 1","type": "event name","time": "1980-01-01T10:28:00.000000",
        "attributes": [{"name":"attr_name","value":"90"}],
        "relationships": [{"objectId": "object_1","qualifier": ""}]}
    """

    tracking_results = pd.read_pickle(session_path + video_session_id + "/tracking_results.pkl") #load results of current session
    amount_of_frames = tracking_results['frame_number'].max() #get number of frames
    print(f"video_event_trigger_algorithm_standard: amount of frames: {amount_of_frames}")
    start_time = time.time()
    print(f"Start time: {start_time}")

    events_list = []
    source_attr = {"name":"event_source","value":"video"}
    source_attr_detail = {"name":"event_source_file","value":video_session_id}
    rule_number = 0
    for rule in rules:
        rule_number += 1
        print(f"Starting rule {rule_number} of {len(rules)}")
        objectID1, objectID2, activity_name = rule[0], rule[1], rule[2]
        state_previous_frame_overlap = False
        for frame in range(1, amount_of_frames+1): #iterate over frames
            print(f"Rule {rule_number}/{len(rules)}. Frame {frame}/{amount_of_frames}")
            all_objIDs_in_frame = tracking_results.loc[(tracking_results["frame_number"] == frame), "object_id_manual"].tolist()
            if objectID1 in all_objIDs_in_frame and objectID2 in all_objIDs_in_frame:
                #print(f"obj1 {objectID1} and obj2 {objectID2}")
                bb_obj_1 = tracking_results.loc[(tracking_results["frame_number"] == frame) & (tracking_results["object_id_manual"] == objectID1), "bounding_box"].values[0]
                bb_obj_2 = tracking_results.loc[(tracking_results["frame_number"] == frame) & (tracking_results["object_id_manual"] == objectID2), "bounding_box"].values[0]
                if bounding_boxes_overlapping(bb_obj_1,bb_obj_2):
                    if state_previous_frame_overlap == False:
                        #event
                        curr_event = {}
                        #attributes
                        curr_event_attr_list = []
                        #curr_attr = {"name": curr_attr_name, "value": curr_attr_value}
                        #curr_event_attr_list.append(curr_attr)
                        curr_event_attr_list.append(source_attr) #add once
                        curr_event_attr_list.append(source_attr_detail) #add once
                        #relations to objects
                        curr_event_rel_list = [{"objectId": objectID1, "qualifier": ""}, {"objectId": objectID2, "qualifier": ""}]
                        #event data
                        curr_event["id"] = activity_name + "_" + str(uuid.uuid4())
                        curr_event["type"] = activity_name
                        curr_event["time"] = get_frame_timestamp(video_start_timestamp_str, video_fps, frame)
                        curr_event["attributes"] = curr_event_attr_list
                        curr_event["relationships"] = curr_event_rel_list
                        events_list.append(curr_event)
                    state_previous_frame_overlap = True
                else:
                    state_previous_frame_overlap = False

    end_time = time.time()
    print(f"End time: {end_time}")
    print("Total duration (H:M:S):", time.strftime('%H:%M:%S', time.gmtime(end_time-start_time)))
    return events_list

def video_event_trigger_algorithm_custom(tracking_results_df_path, video_start_timestamp_str):
    """
    video_event_trigger_algorithm_custom is a custom algorithm that defines events based on individual/use case based needs.
        The algorithm can thus be defined individually by a user. A dataframe containing all events is returned.
        Ideas: an event could be defined when the bounding boxes of two detected/tracked objects start to overlap and another event when the bounding boxes stop overlapping.
        Another idea is to limit the event definition to specific object classes, e.g., an event could be only triggered when the bounding boxes of tracked objects
        with the object classes 'employee' and 'item' start overlapping.
        Triggering events could also be limited to specific areas in the frame, or when bounding boxes are not yet overlapping but close to each other.

    :tracking_results_df_path: The path to the current session folder including a '/' at the end.
        Note: The results dataframe in this session folder has to be named 'tracking_results.pkl'
    :video_start_timestamp: The exact timestamp when the video started, i.e., the timestamp of frame 1. Format: '1980-01-01T10:28:00.000000+0000'
    :return: Returns a dataframe containing all defined events with event ID, timestamp, activity, related objects, and event attributes
        columns: 'ocel:eid', 'ocel:timestamp', 'ocel:activity', 'ocel:omap', 'ocel:vmap'
        Example row: "create order 1", "1980-01-01T10:25:00.000000+0000", "create-order", ["order creating employee 2", "item 4"], {"total-items": 1}
    """

    #Custom algorithm based on use case (IT asset management dataset)
    #When two object bounding boxes overlap from the beginning (frame 0) then this is not treated as an event
    tracking_results = pd.read_pickle(tracking_results_df_path + "tracking_results.pkl") #load results of current session
    amount_of_frames = tracking_results['frame_number'].max() #get number of frames
    states_previous_frame = {}
    rolling_number_activities = {}
    events_list = []
    for frame in range(1, amount_of_frames+1): #iterate over frames
        object_ids_in_frame = tracking_results[tracking_results["frame_number"] == frame].object_id_manual.unique() #get all unique objects (object_id_manual) in that frame
        for i, object_1_id in enumerate(object_ids_in_frame): #iterate over objects (manual object IDs)
            for j in range(i+1,len(object_ids_in_frame)): #iterate over all other objects (manual object IDs)
                object_2_id = object_ids_in_frame[j]
                bb_obj_1 = tracking_results.loc[(tracking_results["frame_number"] == frame) & (tracking_results["object_id_manual"] == object_1_id), "bounding_box"].values[0]
                bb_obj_2 = tracking_results.loc[(tracking_results["frame_number"] == frame) & (tracking_results["object_id_manual"] == object_2_id), "bounding_box"].values[0]
                both_objects_dict_key = object_1_id+","+object_2_id
                if bounding_boxes_overlapping(bb_obj_1,bb_obj_2):
                    if states_previous_frame.get(both_objects_dict_key) in (False, None) and frame > 1: #objects did not overlap in previous frame or combo did not exist. Skip first frame
                        curr_activity = object_1_id + " started interaction with " + object_2_id
                        if rolling_number_activities.get(curr_activity) == None:
                            rolling_event_id = 1
                        else:
                            rolling_event_id = rolling_number_activities.get(curr_activity) + 1
                        rolling_number_activities[curr_activity] = rolling_event_id
                        curr_event_id = curr_activity + "_" + str(rolling_event_id)
                        curr_timestamp = get_frame_timestamp(video_start_timestamp_str, video_fps, frame)
                        curr_event_attr = {"event_source": "video"}
                        curr_rel_obj = [object_1_id, object_2_id]
                        events_list.append([curr_event_id, curr_timestamp, curr_activity, curr_rel_obj, curr_event_attr]) #event bb started overlapping
                    states_previous_frame[both_objects_dict_key] = True #Save new bounding box overlapping state for later
                else: #objects not overlapping
                    if states_previous_frame.get(both_objects_dict_key) == True: #objects did overlap in previous frame. Condition =None for first frame since dict={}
                        curr_activity = object_1_id + " ended interaction with " + object_2_id
                        if rolling_number_activities.get(curr_activity) == None:
                            rolling_event_id = 1
                        else:
                            rolling_event_id = rolling_number_activities.get(curr_activity) + 1
                        rolling_number_activities[curr_activity] = rolling_event_id
                        curr_event_id = curr_activity + "_" + str(rolling_event_id)
                        curr_timestamp = get_frame_timestamp(video_start_timestamp_str, video_fps, frame)
                        curr_event_attr = {"event_source": "video"}
                        curr_rel_obj = [object_1_id, object_2_id]
                        events_list.append([curr_event_id, curr_timestamp, curr_activity, curr_rel_obj, curr_event_attr]) #event bb started overlapping
                    states_previous_frame[both_objects_dict_key] = False #Save new bounding box overlapping state for later

    event_df = pd.DataFrame(events_list, columns = ['ocel:eid', 'ocel:timestamp', 'ocel:activity', 'ocel:omap', 'ocel:vmap']) #create final events dataframe

    return event_df

def video_get_objects_df(tracking_results_df_path):
    """
    video_get_objects_df takes the tracking results dataframe and extracts all detected/tracked objects.
        It returns a dataframe containing all the objects with object ID, objetc type, and object attributes. Also the objects that have no relation to an event.

    :tracking_results_df_path: The path to the current session folder including a '/' at the end.
        Note: The results dataframe in this session folder has to be named 'tracking_results.pkl'
    :return: Returns a dataframe containing all detected/tracked objects with attributes. Also the objects that have no relation to an event
    """

    tracking_results = pd.read_pickle(tracking_results_df_path + "tracking_results.pkl") #load results of current session
    tracking_results = tracking_results[['object_id_manual', 'object_class_manual', 'object_attr_manual']] #only keep relevant columns with object ID, object class, and object attributes
    tracking_results.drop_duplicates(subset=['object_id_manual'], inplace=True) #only keep every object (by object_id_manual) once
    tracking_results.rename(columns={'object_id_manual': 'ocel:oid', 'object_class_manual':'ocel:type', 'object_attr_manual': 'ocel:ovmap'}, inplace=True) #rename columns

    #if there is not yet an object source attribute stating that the object was originally found in the process log, add this attribute stating that the source is the video
    for index, row in tracking_results.iterrows():
        obj_attr_dict = row['ocel:ovmap']
        obj_attr_source = obj_attr_dict.get("object_source_original")
        if obj_attr_source == None:
            obj_attr_dict["object_source_original"] = "video" #as a standard attribute save for later that this object initially was found in the video
            tracking_results.at[index, 'ocel:ovmap'] = obj_attr_dict

    return tracking_results


if __name__ == "__main__":
    
