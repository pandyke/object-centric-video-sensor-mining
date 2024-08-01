import tkinter
from customtkinter import filedialog
import tkinter.messagebox
import customtkinter
import os
from datetime import datetime
import pandas as pd
import pm4py
import json
from PIL import Image
from pathlib import Path
import cv2


from process_log_processing import preprocessProcessLog
from process_log_processing import processLogToOCEL
from process_log_processing import objects_events_from_ocel

from video_processing import object_tracking, init_object_tracking, object_labeling_image_prep, object_labeling_post_annotator, manual_object_definer_prep
from video_processing import select_bounding_boxes, create_image_with_bounding_box, manual_object_definer, video_event_trigger_algorithm_standard

from sensor_processing import extract_events_continuous_data, extract_events_discrete_data

from analysis import get_all_event_types, update_or_create_object, get_object_type_and_attributes, get_all_object_ids, get_all_object_types, add_events, get_events_summary

from ocel_utilities import analyzeOCEL, replace_null_qualifier_ocel2

customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("Object-Centric Video and Sensor Mining")
        #self.geometry(f"{1100}x{580}")
        w, h = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry("%dx%d+0+0" % (w*0.8, h*0.8))

        #configure grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        #self.grid_columnconfigure((2, 3), weight=0)
        #self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with menu options
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.sidebar_frame.grid_rowconfigure(8, weight=1)

        customtkinter.CTkLabel(self.sidebar_frame, text="OCVSM", font=customtkinter.CTkFont(size=30, weight="bold")).grid(row=0, column=0, padx=20, pady=(20, 10))

        self.sidebar_button_start = customtkinter.CTkButton(self.sidebar_frame, command=self.sidebar_button_start_click, text="Start")
        self.sidebar_button_start.grid(row=1, column=0, padx=20, pady=10)

        self.sidebar_button_process_log = customtkinter.CTkButton(self.sidebar_frame, command=self.sidebar_button_process_log_click, text="Process Log")
        self.sidebar_button_process_log.grid(row=2, column=0, padx=20, pady=10)
        self.sidebar_button_process_log.configure(state="disabled")

        self.sidebar_button_video_labeler = customtkinter.CTkButton(self.sidebar_frame, command=self.sidebar_button_video_labeler_click, text="Video Object Labeler")
        self.sidebar_button_video_labeler.grid(row=3, column=0, padx=20, pady=10)
        self.sidebar_button_video_labeler.configure(state="disabled")

        self.sidebar_button_object_area = customtkinter.CTkButton(self.sidebar_frame, command=self.sidebar_button_object_area_click, text="Video Object Area Definer")
        self.sidebar_button_object_area.grid(row=4, column=0, padx=20, pady=10)
        self.sidebar_button_object_area.configure(state="disabled")

        self.sidebar_button_video_event = customtkinter.CTkButton(self.sidebar_frame, command=self.sidebar_button_video_event_click, text="Video Event Extractor")
        self.sidebar_button_video_event.grid(row=5, column=0, padx=20, pady=10)
        self.sidebar_button_video_event.configure(state="disabled")

        self.sidebar_button_sensor = customtkinter.CTkButton(self.sidebar_frame, command=self.sidebar_button_sensor_click, text="Sensor")
        self.sidebar_button_sensor.grid(row=6, column=0, padx=20, pady=10)
        self.sidebar_button_sensor.configure(state="disabled")

        self.sidebar_button_analysis = customtkinter.CTkButton(self.sidebar_frame, command=self.sidebar_button_analysis_click, text="Analysis")
        self.sidebar_button_analysis.grid(row=7, column=0, padx=20, pady=10)
        self.sidebar_button_analysis.configure(state="disabled")

        customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w").grid(row=9, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"], command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=10, column=0, padx=20, pady=(10, 10))
        self.appearance_mode_optionemenu.set("Dark") #default value

        #self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="Scaling:", anchor="w")
        #self.scaling_label.grid(row=9, column=0, padx=20, pady=(10, 0))
        #self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"], command=self.change_scaling_event)
        #self.scaling_optionemenu.grid(row=12, column=0, padx=20, pady=(10, 20))
        #self.scaling_optionemenu.set("100%") #default value


        #create main frame to show different pages
        self.main_frame = customtkinter.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, padx=10, pady=10, sticky='news')
        self.main_frame.grid_columnconfigure(1, weight=2)
        #self.main_frame.grid_columnconfigure((0,2), weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        #create the start frame
        self.start_f = customtkinter.CTkFrame(self.main_frame)
        self.start_f.grid(row=0, column=1, sticky='n', padx=10, pady=10)
        customtkinter.CTkLabel(self.start_f, text='Object-Centric Video and Sensor Mining', font=customtkinter.CTkFont(size=20, weight="bold")).grid(row=0, column=1, padx=20, pady=(20, 10))
        customtkinter.CTkLabel(self.start_f, text='This tool aims to combine unstructured video- and sensor data with structured process logs for object-centric process mining.').grid(row=1, column=1, padx=20, pady=(20, 10))
        customtkinter.CTkLabel(self.start_f, text='Please start a new session or load an existing session.').grid(row=2, column=1, padx=20, pady=(20, 5))
        customtkinter.CTkButton(self.start_f, text='Start new session', command=self.start_new_session).grid(row=3, column=1, padx=20, pady=(5, 5))
        customtkinter.CTkButton(self.start_f, text='Load existing session', command=self.load_session).grid(row=4, column=1, padx=20, pady=(5, 10))
        self.start_feedback_text = customtkinter.CTkLabel(self.start_f, text='', text_color="red")
        self.start_feedback_text.grid(row=5, column=1, padx=20, pady=(5, 5))

        #create the process log frame
        self.process_log_f = customtkinter.CTkFrame(self.main_frame)
        self.process_log_f.grid(row=0, column=1, sticky='n', padx=10, pady=10)
        self.process_log_f_1 = customtkinter.CTkFrame(self.process_log_f)
        self.process_log_f_1.grid(row=0, column=0, columnspan=3, sticky='n', padx=10, pady=10)
        self.process_log_f_2 = customtkinter.CTkFrame(self.process_log_f)
        self.process_log_f_2.grid(row=1, column=0, columnspan=2, sticky='n', padx=10, pady=10)
        self.process_log_f_3 = customtkinter.CTkFrame(self.process_log_f)
        self.process_log_f_3.grid(row=1, column=2, sticky='n', padx=10, pady=10)

        customtkinter.CTkLabel(self.process_log_f_1, text='Process Log', font=customtkinter.CTkFont(size=20, weight="bold")).grid(row=0, column=1, columnspan=5, padx=20, pady=(20, 10))
        customtkinter.CTkLabel(self.process_log_f_1, wraplength=1200, text='In the first step process logs can be loaded that are later enhanced with video and sensor data. ' + 
            'The process logs can either be standard ocel logs or in a table format that can be preprocessed. Objects and events are extracted from the process logs. ' +
            'It is possible to load multiple logs, the objects and events will then be appended to the existing ones.').grid(row=1, column=1, columnspan=5, padx=20, pady=(20, 10))
        customtkinter.CTkLabel(self.process_log_f_2, text='From Table Format', font=customtkinter.CTkFont(size=20, weight="bold")).grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 10))
        #preprocessing file
        customtkinter.CTkLabel(self.process_log_f_2, wraplength=380, text="Select a file in table format that is then automatically preprocessed. " +
            "In the code the function 'preprocessProcessLog' in the file 'process_log_processing.py' " + 
            "can be customized for individual needs. The preprocessed file is saved to the subfolder " +
            "'process_logs_preprocessed' in the current session folder.").grid(row=1, column=0, padx=20, pady=(20, 10))
        customtkinter.CTkButton(self.process_log_f_2, text='Preprocess file', command=self.process_log_preprocess_file).grid(row=2, column=0, padx=20, pady=(20, 10))
        self.process_log_feedback_text_1 = customtkinter.CTkLabel(self.process_log_f_2, text='', text_color="green")
        self.process_log_feedback_text_1.grid(row=3, column=0, padx=20, pady=(5, 5))
        #Extracting objects and events from preprocessed file
        customtkinter.CTkLabel(self.process_log_f_2, wraplength=380, text='Select a preprocessed file from which objects and events are then automatically extracted ' +
            'and appended to existing objects and events from previously loaded process logs.').grid(row=1, column=1, padx=20, pady=(20, 10))
        customtkinter.CTkButton(self.process_log_f_2, text='Extract from preprocessed file', command=self.prepr_process_log_extract_events_objects).grid(
                                                                                                                            row=2, column=1, padx=20, pady=(20, 10))
        self.process_log_feedback_text_2 = customtkinter.CTkLabel(self.process_log_f_2, text='', text_color="green")
        self.process_log_feedback_text_2.grid(row=3, column=1, padx=20, pady=(5, 5))
        #Extracting objects and events from ocel file
        customtkinter.CTkLabel(self.process_log_f_3, text='From OCEL', font=customtkinter.CTkFont(size=20, weight="bold")).grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 10))
        customtkinter.CTkLabel(self.process_log_f_3, wraplength=380, text='Select a valid ocel file from which objects and events are then automatically extracted ' +
            'and appended to existing objects and events from previously loaded process logs.').grid(row=1, column=0, padx=20, pady=(20, 10))
        customtkinter.CTkButton(self.process_log_f_3, text='Extract from ocel file', command=self.ocel_file_extract_events_objects).grid(
                                                                                                                            row=2, column=0, padx=20, pady=(20, 10))
        self.process_log_feedback_text_3 = customtkinter.CTkLabel(self.process_log_f_3, text='', text_color="green")
        self.process_log_feedback_text_3.grid(row=3, column=0, padx=20, pady=(5, 5))


        #create the video labeler frame
        self.video_labeler_f = customtkinter.CTkFrame(self.main_frame)
        self.video_labeler_f.grid(row=0, column=1, sticky='n', padx=10, pady=10)
        self.video_labeler_f_header = customtkinter.CTkLabel(self.video_labeler_f, text='Object Tracking and Labeling', font=customtkinter.CTkFont(size=20, weight="bold"))
        self.video_labeler_f_header.grid(row=0, column=6, columnspan=7, sticky='n', padx=20, pady=(20, 10))
        self.video_labeler_f_text_descr = customtkinter.CTkLabel(self.video_labeler_f, text='Run object tracking and select a video or load existing object tracking results.')
        self.video_labeler_f_text_descr.grid(row=1, column=6,columnspan=7, sticky='n', padx=20, pady=(20, 10))
        self.video_labeler_f_button_runObjTracking = customtkinter.CTkButton(self.video_labeler_f, text='Run Object Tracking', command=self.run_object_tracking)
        self.video_labeler_f_button_runObjTracking.grid(row=2, column=8, padx=20, pady=(20, 0))
        self.video_labeler_f_feedback_runObjTracking = customtkinter.CTkLabel(self.video_labeler_f, text='', text_color="green")
        self.video_labeler_f_feedback_runObjTracking.grid(row=3, column=8, padx=20, pady=(5, 0))
        self.video_labeler_f_button_loadObjTracking = customtkinter.CTkButton(self.video_labeler_f, text='Load existing Object Tracking results', command=self.load_object_tracking)
        self.video_labeler_f_button_loadObjTracking.grid(row=2, column=10, padx=20, pady=(20, 0))
        self.video_labeler_f_feedback_loadObjTracking = customtkinter.CTkLabel(self.video_labeler_f, text='', text_color="green")
        self.video_labeler_f_feedback_loadObjTracking.grid(row=3, column=10, padx=20, pady=(5, 0))

        #frame in frame (for actual labeling)
        self.video_labeler_labeling_f = customtkinter.CTkFrame(self.video_labeler_f)
        #self.video_labeler_labeling_f.grid(row=4, column=1, columnspan=18, sticky='n', padx=10, pady=10) #grid when this frame should be shown. Otherwise: self.video_labeler_labeling_f.grid_remove()
        self.video_labelling_f_button_ignoreObj = customtkinter.CTkSwitch(self.video_labeler_labeling_f, text='Ignore Object', command=self.ignore_object)
        self.video_labelling_f_button_ignoreObj.grid(row=1, column=1, padx=20, pady=(20, 10))
        self.video_labelling_f_button_PrevObj = customtkinter.CTkButton(self.video_labeler_labeling_f, text='Previous Object', command=self.previous_object)
        self.video_labelling_f_button_PrevObj.grid(row=1, column=3, padx=20, pady=(20, 10))
        self.video_labelling_f_curr_obj_id_number_label = customtkinter.CTkLabel(self.video_labeler_labeling_f, text='', text_color="white")
        self.video_labelling_f_curr_obj_id_number_label.grid(row=1, column=4, padx=20, pady=(20, 10))

        self.video_labelling_f_button_nextObj = customtkinter.CTkButton(self.video_labeler_labeling_f, text='Next Object', command=self.next_object)
        self.video_labelling_f_button_nextObj.grid(row=1, column=5, padx=20, pady=(20, 10))
        self.video_labelling_f_button_finish = customtkinter.CTkButton(self.video_labeler_labeling_f, text='Finish Labeling', command=self.finish_labeling)
        self.video_labelling_f_button_finish.grid(row=1, column=7, padx=20, pady=(20, 10))

        #img = customtkinter.CTkImage(light_image=Image.open('data/example_labeling_img.jpg'), dark_image=Image.open('data/example_labeling_img.jpg'), size=(1152,648)) #WidthxHeight
        #self.video_labelling_f_image = customtkinter.CTkLabel(self.video_labeler_labeling_f, image = img)
        #self.video_labelling_f_image.grid(row=2, rowspan=48, column=1, columnspan=7, padx=20, pady=(20, 10))

        self.video_labelling_f_objID_label = customtkinter.CTkLabel(self.video_labeler_labeling_f, text='Object ID')
        self.video_labelling_f_objID_label.grid(row=2, column=8, columnspan=2, padx=20, pady=(0, 0))
        dropdwon_objID_init_val = customtkinter.StringVar(value="")  # set initial value
        self.video_labelling_f_objID_dropd = customtkinter.CTkComboBox(self.video_labeler_labeling_f, state="normal", variable=dropdwon_objID_init_val,
                                                                 command=self.dropdwon_objID_clicked, values=(""), width=250)
        self.video_labelling_f_objID_dropd.grid(row=3, column=8, columnspan=2, padx=20, pady=(0, 20))

        self.video_labelling_f_objtype_label = customtkinter.CTkLabel(self.video_labeler_labeling_f, text='Object Type')
        self.video_labelling_f_objtype_label.grid(row=5, column=8, columnspan=2, padx=20, pady=(0, 0))
        dropdwon_objType_init_val = customtkinter.StringVar(value="")  # set initial value
        self.video_labelling_f_objType_dropd = customtkinter.CTkComboBox(self.video_labeler_labeling_f, state="normal",
                                                                         variable=dropdwon_objType_init_val, values=(""), width=250)
        self.video_labelling_f_objType_dropd.grid(row=6, column=8, columnspan=2, padx=20, pady=(0, 20))

        self.video_labelling_f_objAttr_label = customtkinter.CTkLabel(self.video_labeler_labeling_f, text='Object Attributes')
        self.video_labelling_f_objAttr_label.grid(row=8, column=8, columnspan=2, padx=20, pady=(0, 0))
        self.video_labelling_f_objAttr_name_label = customtkinter.CTkLabel(self.video_labeler_labeling_f, text='Name')
        self.video_labelling_f_objAttr_name_label.grid(row=9, column=8, padx=20, pady=(0, 0))
        self.video_labelling_f_objAttr_value_label = customtkinter.CTkLabel(self.video_labeler_labeling_f, text='Value')
        self.video_labelling_f_objAttr_value_label.grid(row=9, column=9, padx=20, pady=(0, 0))

        self.video_labelling_f_objAttr_names_list = customtkinter.CTkTextbox(self.video_labeler_labeling_f)
        #self.video_labelling_f_objAttr_names_list.insert("end", "test" + "\n") #index is 'line.character'
        #self.video_labelling_f_objAttr_names_list.insert("end", "test2" + "\n") #index is 'line.character'
        self.video_labelling_f_objAttr_names_list.grid(row=10, column=8, padx=1, pady=(0, 10))

        self.video_labelling_f_objAttr_values_list = customtkinter.CTkTextbox(self.video_labeler_labeling_f)
        #self.video_labelling_f_objAttr_values_list.insert("end", "test value" + "\n") #index is 'line.character'
        #self.video_labelling_f_objAttr_values_list.insert("end", "test value 2" + "\n") #index is 'line.character'
        self.video_labelling_f_objAttr_values_list.grid(row=10, column=9, padx=0, pady=(0, 10))




        #create the video object area definer frame
        self.video_obj_area_f = customtkinter.CTkFrame(self.main_frame)
        self.video_obj_area_f.grid(row=0, column=1, sticky='n', padx=10, pady=10)
        self.video_obj_area_f_header = customtkinter.CTkLabel(self.video_obj_area_f, text='Video Object Area Definer', font=customtkinter.CTkFont(size=20, weight="bold"))
        self.video_obj_area_f_header.grid(row=0, column=8, columnspan=5, padx=20, pady=(20, 10))
        self.video_obj_area_f_text_descr = customtkinter.CTkLabel(self.video_obj_area_f, text='Here static objects can be manually defined by selecting bounding boxes in the frame.')
        self.video_obj_area_f_text_descr.grid(row=1, column=8, columnspan=5, padx=20, pady=(20, 10))
        self.video_obj_area_f_button_select_tracking_res = customtkinter.CTkButton(self.video_obj_area_f, text='Load Object Tracking Results',
                                                                                   command=self.area_definer_select_tracking_results)
        self.video_obj_area_f_button_select_tracking_res.grid(row=2, column=10, padx=20, pady=(20, 10))
        self.video_obj_area_f_sel_trackResults_feedback = customtkinter.CTkLabel(self.video_obj_area_f, text='')
        self.video_obj_area_f_sel_trackResults_feedback.grid(row=3, column=10, padx=20, pady=(5, 0))

        #frame in frame (for actual area definer)
        self.video_obj_area_f_definer = customtkinter.CTkFrame(self.video_obj_area_f)
        #self.video_obj_area_f_definer.grid(row=4, column=1, columnspan=18, sticky='n', padx=10, pady=10) #grid when this frame should be shown. Otherwise: self.video_labeler_labeling_f.grid_remove()
        self.video_obj_area_f_button_select_bb = customtkinter.CTkButton(self.video_obj_area_f_definer, text='Select Bounding Box of new Object', command=self.select_bounding_box)
        self.video_obj_area_f_button_select_bb.grid(row=1, column=2, padx=20, pady=(20, 10))
        self.video_obj_area_f_button_select_bb_feedback = customtkinter.CTkLabel(self.video_obj_area_f_definer, text='')
        self.video_obj_area_f_button_select_bb_feedback.grid(row=2, column=2, padx=20, pady=(0, 0))
        self.video_obj_area_f_button_save = customtkinter.CTkButton(self.video_obj_area_f_definer, text='Save defined Object', command=self.save_defined_object)
        self.video_obj_area_f_button_save.grid(row=1, column=4, padx=20, pady=(20, 10))
        self.video_obj_area_f_button_save_feedback = customtkinter.CTkLabel(self.video_obj_area_f_definer, text='')
        self.video_obj_area_f_button_save_feedback.grid(row=2, column=4, padx=20, pady=(0, 0))
        self.video_obj_area_f_button_finish = customtkinter.CTkButton(self.video_obj_area_f_definer, text='Finish Object Definer', command=self.finish_object_definer)
        self.video_obj_area_f_button_finish.grid(row=1, column=7, padx=20, pady=(20, 10))

        #img = customtkinter.CTkImage(light_image=Image.open('data/example_labeling_img.jpg'), dark_image=Image.open('data/example_labeling_img.jpg'), size=(1152,648)) #WidthxHeight
        #self.video_obj_area_f_image = customtkinter.CTkLabel(self.video_obj_area_f_definer, image = img)
        #self.video_obj_area_f_image.grid(row=3, rowspan=47, column=1, columnspan=7, padx=20, pady=(20, 10))

        self.video_obj_area_f_objID_label = customtkinter.CTkLabel(self.video_obj_area_f_definer, text='Object ID')
        self.video_obj_area_f_objID_label.grid(row=4, column=8, columnspan=2, padx=20, pady=(0, 0))
        self.video_obj_area_f_objID_dropd = customtkinter.CTkTextbox(self.video_obj_area_f_definer, height=25, width=250)
        self.video_obj_area_f_objID_dropd.grid(row=5, column=8, columnspan=2, padx=20, pady=(0, 20))
        self.video_obj_area_f_objtype_label = customtkinter.CTkLabel(self.video_obj_area_f_definer, text='Object Type')
        self.video_obj_area_f_objtype_label.grid(row=7, column=8, columnspan=2, padx=20, pady=(0, 0))
        dropdwon_objType_init_val = customtkinter.StringVar(value="")  # set initial value
        self.video_obj_area_f_objType_dropd = customtkinter.CTkComboBox(self.video_obj_area_f_definer, state="normal",
                                                                         variable=dropdwon_objType_init_val, command=self.dropdwon_objType_clicked, values=(""), width=250)
        self.video_obj_area_f_objType_dropd.grid(row=8, column=8, columnspan=2, padx=20, pady=(0, 20))

        self.video_obj_area_f_objAttr_label = customtkinter.CTkLabel(self.video_obj_area_f_definer, text='Object Attributes')
        self.video_obj_area_f_objAttr_label.grid(row=10, column=8, columnspan=2, padx=20, pady=(0, 0))
        self.video_obj_area_f_objAttr_name_label = customtkinter.CTkLabel(self.video_obj_area_f_definer, text='Name')
        self.video_obj_area_f_objAttr_name_label.grid(row=11, column=8, padx=20, pady=(0, 0))
        self.video_obj_area_f_objAttr_value_label = customtkinter.CTkLabel(self.video_obj_area_f_definer, text='Value')
        self.video_obj_area_f_objAttr_value_label.grid(row=11, column=9, padx=20, pady=(0, 0))

        self.video_obj_area_f_objAttr_names_list = customtkinter.CTkTextbox(self.video_obj_area_f_definer)
        #self.video_obj_area_f_objAttr_names_list.insert("end", "test" + "\n") #index is 'line.character'
        #self.video_obj_area_f_objAttr_names_list.insert("end", "test2" + "\n") #index is 'line.character'
        self.video_obj_area_f_objAttr_names_list.grid(row=12, column=8, padx=1, pady=(0, 10))

        self.video_obj_area_f_objAttr_values_list = customtkinter.CTkTextbox(self.video_obj_area_f_definer)
        #self.video_obj_area_f_objAttr_values_list.insert("end", "test value" + "\n") #index is 'line.character'
        #self.video_obj_area_f_objAttr_values_list.insert("end", "test value 2" + "\n") #index is 'line.character'
        self.video_obj_area_f_objAttr_values_list.grid(row=12, column=9, padx=0, pady=(0, 10))


        #create the video event extractor frame
        self.video_event_f = customtkinter.CTkFrame(self.main_frame)
        self.video_event_f.grid(row=0, column=1, sticky='n', padx=10, pady=10)
        self.video_event_f_header = customtkinter.CTkLabel(self.video_event_f, text='Video Event Extractor', font=customtkinter.CTkFont(size=20, weight="bold"))
        self.video_event_f_header.grid(row=0, column=8, columnspan=5, padx=20, pady=(20, 10))
        self.video_event_f_text_descr = customtkinter.CTkLabel(self.video_event_f, text='Here events can be extracted from prepared video data.')
        self.video_event_f_text_descr.grid(row=1, column=8, columnspan=5, padx=20, pady=(20, 10))
        self.video_event_f_button_select_tracking_res = customtkinter.CTkButton(self.video_event_f, text='Load Object Tracking Results',
                                                                                   command=self.event_extractor_select_tracking_results)
        self.video_event_f_button_select_tracking_res.grid(row=2, column=10, padx=20, pady=(20, 0))
        self.video_event_f_sel_trackResults_feedback = customtkinter.CTkLabel(self.video_event_f, text='')
        self.video_event_f_sel_trackResults_feedback.grid(row=3, column=10, padx=20, pady=(0, 0))

        #frame in frame (for actual event extractor)
        self.video_event_f_extractor = customtkinter.CTkFrame(self.video_event_f)
        #self.video_event_f_extractor.grid(row=4, column=1, columnspan=18, sticky='n', padx=10, pady=10) #grid when this frame should be shown. Otherwise: self.video_event_f_extractor.grid_remove()
        self.video_event_f_extractor_timestamp_label = customtkinter.CTkLabel(self.video_event_f_extractor,
                                                                              text="Start Timestamp of the Video \n Format '1970-01-01T00:00:00.000000'")
        self.video_event_f_extractor_timestamp_label.grid(row=1, column=1, padx=20, pady=(10, 0))
        self.video_event_f_extractor_timestamp_input = customtkinter.CTkTextbox(self.video_event_f_extractor, height=25, width=250, text_color="grey")
        self.video_event_f_extractor_timestamp_input.grid(row=2, column=1, padx=20, pady=(0, 0))
        self.video_event_f_extractor_button_extractEvents = customtkinter.CTkButton(self.video_event_f_extractor, text='Extract Events', command=self.extract_events)
        self.video_event_f_extractor_button_extractEvents.grid(row=1, column=2, padx=20, pady=(10, 0))
        self.video_event_f_extractor_button_extractEvents_feedback = customtkinter.CTkLabel(self.video_event_f_extractor, text='')
        self.video_event_f_extractor_button_extractEvents_feedback.grid(row=2, column=2, padx=20, pady=(0, 0))
        self.video_event_f_extractor_button_finish_extractor = customtkinter.CTkButton(self.video_event_f_extractor, text='Finish (go to Sensor)',
                                                                                       command=self.finish_extract_events)
        self.video_event_f_extractor_button_finish_extractor.grid(row=1, column=3, padx=20, pady=(10, 0))

        self.video_event_f_extractor_button_load_additional_video = customtkinter.CTkButton(self.video_event_f_extractor, text='Load additional video (go to Object Labeler)',
                                                                                       command=self.load_additional_video)
        self.video_event_f_extractor_button_load_additional_video.grid(row=2, column=3, padx=20, pady=(20, 0))
        self.video_event_f_extractor_rules_label = customtkinter.CTkLabel(self.video_event_f_extractor,
            text="Add all event rules and then extract events. An event rule defines that, if the bounding boxes of 'Object ID 1' and 'Object ID 2' start overlapping, then the event 'Activity Name' occurs.")
        self.video_event_f_extractor_rules_label.grid(row=3, column=1, columnspan=3, padx=20, pady=(60, 10))
        self.video_event_f_extractor_button_add_rule = customtkinter.CTkButton(self.video_event_f_extractor, text='Add Rule',
                                                                                       command=self.add_rule)
        self.video_event_f_extractor_button_add_rule.grid(row=4, column=2, padx=20, pady=(0, 10))
        self.video_event_f_extractor_objID1_label = customtkinter.CTkLabel(self.video_event_f_extractor, text='Object ID 1')
        self.video_event_f_extractor_objID1_label.grid(row=5, column=1, padx=20, pady=(0, 0))
        self.video_event_f_extractor_objID2_label = customtkinter.CTkLabel(self.video_event_f_extractor, text='Object ID 2')
        self.video_event_f_extractor_objID2_label.grid(row=5, column=2, padx=20, pady=(0, 0))
        self.video_event_f_extractor_eventType_label = customtkinter.CTkLabel(self.video_event_f_extractor, text='Activity Name')
        self.video_event_f_extractor_eventType_label.grid(row=5, column=3, padx=20, pady=(0, 0))
        self.video_event_f_extractor_objID1_dropd = customtkinter.CTkComboBox(self.video_event_f_extractor, state="normal",
                                                                         variable=customtkinter.StringVar(value=""), values=(""), width=250)
        self.video_event_f_extractor_objID1_dropd.grid(row=6, column=1, padx=20, pady=(0, 20))
        self.video_event_f_extractor_objID2_dropd = customtkinter.CTkComboBox(self.video_event_f_extractor, state="normal",
                                                                         variable=customtkinter.StringVar(value=""), values=(""), width=250)
        self.video_event_f_extractor_objID2_dropd.grid(row=6, column=2, padx=20, pady=(0, 20))
        self.video_event_f_extractor_eventType_input = customtkinter.CTkTextbox(self.video_event_f_extractor, height=25, width=250)
        self.video_event_f_extractor_eventType_input.grid(row=6, column=3, padx=20, pady=(0, 20))

        self.video_event_f_extractor_objID1_list = customtkinter.CTkTextbox(self.video_event_f_extractor, height=450)
        #self.video_event_f_extractor_objID1_list.insert("end", "test" + "\n") #index is 'line.character'
        self.video_event_f_extractor_objID1_list.grid(row=7, column=1, padx=1, pady=(0, 10))
        self.video_event_f_extractor_objID2_list = customtkinter.CTkTextbox(self.video_event_f_extractor, height=450)
        self.video_event_f_extractor_objID2_list.grid(row=7, column=2, padx=1, pady=(0, 10))
        self.video_event_f_extractor_eventType_list = customtkinter.CTkTextbox(self.video_event_f_extractor, height=450)
        self.video_event_f_extractor_eventType_list.grid(row=7, column=3, padx=1, pady=(0, 10))


        #create the sensor frame
        self.sensor_f = customtkinter.CTkFrame(self.main_frame)
        self.sensor_f.grid(row=0, column=1, sticky='n', padx=10, pady=10)
        self.sensor_f_header = customtkinter.CTkLabel(self.sensor_f, text='Sensor Data', font=customtkinter.CTkFont(size=20, weight="bold"))
        self.sensor_f_header.grid(row=0, column=5, padx=20, pady=(20, 10))
        self.sensor_f_text_descr = customtkinter.CTkLabel(self.sensor_f, text='Load a .csv file containing discrete or continuous sensor data.')
        self.sensor_f_text_descr.grid(row=1, column=5, padx=20, pady=(20, 10))
        self.sensor_f_button_load_sensor_file = customtkinter.CTkButton(self.sensor_f, text='Load Sensor .csv', command=self.load_sensor_file)
        self.sensor_f_button_load_sensor_file.grid(row=2, column=5, padx=20, pady=(20, 10))
        self.sensor_f_button_load_sensor_file_feedback = customtkinter.CTkLabel(self.sensor_f, text='')
        self.sensor_f_button_load_sensor_file_feedback.grid(row=3, column=5, padx=20, pady=(0, 0))
        self.sensor_f_button_finish_sensor_step = customtkinter.CTkButton(self.sensor_f, text='Finish Sensor Step', command=self.finish_sensor_step)
        self.sensor_f_button_finish_sensor_step.grid(row=2, column=8, padx=5, pady=(20, 10))


        #frame in frame (for sensor details)
        self.sensor_f_details_f = customtkinter.CTkFrame(self.sensor_f)
        #self.sensor_f_details_f.grid(row=4, column=1, columnspan=8, sticky='n', padx=10, pady=10) #grid when this frame should be shown. Otherwise: self.sensor_f_details_f.grid_remove()
        sensor_f_details_f_descr_text = "Here details on the sensor can be added. Please provide the names of the columns that include the timestamps and the sensor values."
        self.sensor_f_details_f_descr = customtkinter.CTkLabel(self.sensor_f_details_f, text=sensor_f_details_f_descr_text)
        self.sensor_f_details_f_descr.grid(row=1, column=1, columnspan=10, padx=20, pady=(5,20))
        self.sensor_f_details_f_discrCont_switch_descr = customtkinter.CTkLabel(self.sensor_f_details_f, text="Continuous / Discrete Data")
        self.sensor_f_details_f_discrCont_switch_descr.grid(row=2, column=1, padx=(5,20), pady=(0,0))
        self.sensor_f_details_f_discrCont_switch = customtkinter.CTkSwitch(self.sensor_f_details_f, text='Discrete Data', command=self.discr_cont_switch)
        self.sensor_f_details_f_discrCont_switch.grid(row=3, column=1, padx=(5,20), pady=(0,5))
        self.sensor_f_details_f_time_label = customtkinter.CTkLabel(self.sensor_f_details_f, text="Timestamp Column")
        self.sensor_f_details_f_time_label.grid(row=2, column=2, padx=(0,5), pady=(0,5), sticky='e')
        self.sensor_f_details_f_time_textbox = customtkinter.CTkTextbox(self.sensor_f_details_f, height=25, width=200)
        self.sensor_f_details_f_time_textbox.grid(row=2, column=3, padx=(0,5), pady=(0,5))
        self.sensor_f_details_f_time_label_feedback = customtkinter.CTkLabel(self.sensor_f_details_f, text="")
        self.sensor_f_details_f_time_label_feedback.grid(row=3, column=2, padx=(0,5), pady=(0,5), sticky='e')
        self.sensor_f_details_f_value_label = customtkinter.CTkLabel(self.sensor_f_details_f, text="Sensor Values Column")
        self.sensor_f_details_f_value_label.grid(row=2, column=4, padx=(0,5), pady=(0,5), sticky='e')
        self.sensor_f_details_f_value_textbox = customtkinter.CTkTextbox(self.sensor_f_details_f, height=25, width=200)
        self.sensor_f_details_f_value_textbox.grid(row=2, column=5, padx=(0,5), pady=(0,5))
        sensor_f_details_f_object_label_text = "Select existing object that the sensor is a part of or create a new object for the sensor. Also related objects can be defined."
        self.sensor_f_details_f_object_label = customtkinter.CTkLabel(self.sensor_f_details_f, text=sensor_f_details_f_object_label_text)
        self.sensor_f_details_f_object_label.grid(row=4, column=1, columnspan=10, padx=(5,20), pady=(40, 20))
        self.sensor_f_details_f_objID_label = customtkinter.CTkLabel(self.sensor_f_details_f, text="Object ID")
        self.sensor_f_details_f_objID_label.grid(row=6, column=1, padx=(0,5), pady=(0,5), sticky='e')
        self.sensor_f_details_f_objID_dropd = customtkinter.CTkComboBox(self.sensor_f_details_f, state="normal", command=self.sensor_dropd_objID_clicked,
                                                                         variable=customtkinter.StringVar(value=""), values=(""), width=200)
        self.sensor_f_details_f_objID_dropd.grid(row=6, column=2, padx=(0,20), pady=(0,5))
        self.sensor_f_details_f_objType_label = customtkinter.CTkLabel(self.sensor_f_details_f, text="Object Type")
        self.sensor_f_details_f_objType_label.grid(row=7, column=1, padx=(0,5), pady=(0,5), sticky='e')
        self.sensor_f_details_f_objType_dropd = customtkinter.CTkComboBox(self.sensor_f_details_f, state="normal",
                                                                         variable=customtkinter.StringVar(value=""), values=(""), width=200)
        self.sensor_f_details_f_objType_dropd.grid(row=7, column=2, padx=(0,20), pady=(0,5))
        self.sensor_f_details_f_attr_label = customtkinter.CTkLabel(self.sensor_f_details_f, text="Object Attributes")
        self.sensor_f_details_f_attr_label.grid(row=5, column=3, columnspan=2, padx=(0,5), pady=(0, 1))
        self.sensor_f_details_f_attr_name_label = customtkinter.CTkLabel(self.sensor_f_details_f, text="Name")
        self.sensor_f_details_f_attr_name_label.grid(row=6, column=3, padx=(0,5), pady=(0,5))
        self.sensor_f_details_f_attr_name_list = customtkinter.CTkTextbox(self.sensor_f_details_f, height=125)
        self.sensor_f_details_f_attr_name_list.grid(row=7, column=3, padx=(0,5), pady=(0,5))
        self.sensor_f_details_f_attr_value_label = customtkinter.CTkLabel(self.sensor_f_details_f, text="Value")
        self.sensor_f_details_f_attr_value_label.grid(row=6, column=4, padx=(0,20), pady=(0,5))
        self.sensor_f_details_f_attr_value_list = customtkinter.CTkTextbox(self.sensor_f_details_f, height=125)
        self.sensor_f_details_f_attr_value_list.grid(row=7, column=4, padx=(0,20), pady=(0,5))
        self.sensor_f_details_f_relObj_label = customtkinter.CTkLabel(self.sensor_f_details_f, text="Related Objects")
        self.sensor_f_details_f_relObj_label.grid(row=5, column=5, padx=(0,5), pady=(0,1))
        self.sensor_f_details_f_relObj_dropd = customtkinter.CTkComboBox(self.sensor_f_details_f, state="normal",
                                                                         variable=customtkinter.StringVar(value=""), values=(""), width=200)
        self.sensor_f_details_f_relObj_dropd.grid(row=6, column=5, padx=(0,5), pady=(0,5))
        self.sensor_f_details_f_relObj_list = customtkinter.CTkTextbox(self.sensor_f_details_f, height=125)
        self.sensor_f_details_f_relObj_list.grid(row=7, column=5, padx=(0,5), pady=(0,5))
        self.sensor_f_details_f_AddrelObj_button = customtkinter.CTkButton(self.sensor_f_details_f, text='Add related Object', command=self.add_related_object)
        self.sensor_f_details_f_AddrelObj_button.grid(row=6, column=6, padx=(0,5), pady=(0,5))

        #frame in frame (for continuous sensor data)
        self.sensor_f_continuous_f = customtkinter.CTkFrame(self.sensor_f)
        #self.sensor_f_continuous_f.grid(row=5, column=1, columnspan=8, sticky='n', padx=10, pady=10) #grid when this frame should be shown. Otherwise: self.sensor_f_continuous_f.grid_remove()
        self.sensor_f_continuous_f_descr = customtkinter.CTkLabel(self.sensor_f_continuous_f, text="Continuous Data: Add Rules on how Events are Extracted from the Sensor File.")
        self.sensor_f_continuous_f_descr.grid(row=1, column=1, columnspan=5, padx=(5,5), pady=(5,20))

        self.sensor_f_continuous_f_lastXEntries = customtkinter.CTkLabel(self.sensor_f_continuous_f, text="When change from last X sensor values to now is... [X >= 1]")
        self.sensor_f_continuous_f_lastXEntries.grid(row=2, column=1, padx=(0,20), pady=(0,5))
        self.sensor_f_continuous_f_lastXEntries_greater_text = customtkinter.CTkTextbox(self.sensor_f_continuous_f, height=25, width=200)
        self.sensor_f_continuous_f_lastXEntries_greater_text.grid(row=3, column=1, padx=(0,20), pady=(0,5))
        self.sensor_f_continuous_f_lastXEntries_smaller_text = customtkinter.CTkTextbox(self.sensor_f_continuous_f, height=25, width=200)
        self.sensor_f_continuous_f_lastXEntries_smaller_text.grid(row=4, column=1, padx=(0,20), pady=(0,5))
        self.sensor_f_continuous_f_greater_smaller_label = customtkinter.CTkLabel(self.sensor_f_continuous_f, text="...greater or smaller than...")
        self.sensor_f_continuous_f_greater_smaller_label.grid(row=2, column=2, padx=(0,20), pady=(0,5))
        self.sensor_f_continuous_f_greater_label = customtkinter.CTkLabel(self.sensor_f_continuous_f, text=">")
        self.sensor_f_continuous_f_greater_label.grid(row=3, column=2, padx=(0,20), pady=(0,5))
        self.sensor_f_continuous_f_smaller_label = customtkinter.CTkLabel(self.sensor_f_continuous_f, text="<")
        self.sensor_f_continuous_f_smaller_label.grid(row=4, column=2, padx=(0,20), pady=(0,5))
        self.sensor_f_continuous_f_threshold = customtkinter.CTkLabel(self.sensor_f_continuous_f, text="...threshold...[> 0.0 and < 0.0]")
        self.sensor_f_continuous_f_threshold.grid(row=2, column=3, padx=(0,20), pady=(0,5))
        self.sensor_f_continuous_f_threshold_greater_text = customtkinter.CTkTextbox(self.sensor_f_continuous_f, height=25, width=200)
        self.sensor_f_continuous_f_threshold_greater_text.grid(row=3, column=3, padx=(0,20), pady=(0,5))
        self.sensor_f_continuous_f_threshold_smaller_text = customtkinter.CTkTextbox(self.sensor_f_continuous_f, height=25, width=200)
        self.sensor_f_continuous_f_threshold_smaller_text.grid(row=4, column=3, padx=(0,20), pady=(0,5))
        self.sensor_f_continuous_f_activityName = customtkinter.CTkLabel(self.sensor_f_continuous_f, text="...the activity name is")
        self.sensor_f_continuous_f_activityName.grid(row=2, column=4, padx=(0,20), pady=(0,5))
        self.sensor_f_continuous_f_activityName_greater_text = customtkinter.CTkTextbox(self.sensor_f_continuous_f, height=25, width=200)
        self.sensor_f_continuous_f_activityName_greater_text.grid(row=3, column=4, padx=(0,20), pady=(0,5))
        self.sensor_f_continuous_f_activityName_smaller_text = customtkinter.CTkTextbox(self.sensor_f_continuous_f, height=25, width=200)
        self.sensor_f_continuous_f_activityName_smaller_text.grid(row=4, column=4, padx=(0,20), pady=(0,5))
        self.sensor_f_continuous_f_extract_events_button = customtkinter.CTkButton(self.sensor_f_continuous_f, text='Extract Events', command=self.continuous_extract_events)
        self.sensor_f_continuous_f_extract_events_button.grid(row=3, column=5, padx=(0,5), pady=(0, 5))
        self.sensor_f_continuous_f_extract_events_button_feedback = customtkinter.CTkLabel(self.sensor_f_continuous_f, text="")
        self.sensor_f_continuous_f_extract_events_button_feedback.grid(row=4, column=5, padx=(0,20), pady=(0,5))

        #frame in frame (for discrete sensor data)
        self.sensor_f_discrete_f = customtkinter.CTkFrame(self.sensor_f)
        #self.sensor_f_discrete_f.grid(row=5, column=1, columnspan=8, sticky='n', padx=10, pady=10) #grid when this frame should be shown. Otherwise: self.sensor_f_continuous_f.grid_remove()
        self.sensor_f_discrete_f_descr = customtkinter.CTkLabel(self.sensor_f_discrete_f, text="Discrete Data: Add activity names for the sensor states.")
        self.sensor_f_discrete_f_descr.grid(row=1, column=1, columnspan=3, padx=(5,5), pady=(5,20))
        self.sensor_f_discrete_f_states_label = customtkinter.CTkLabel(self.sensor_f_discrete_f, text="When sensor switches to state...")
        self.sensor_f_discrete_f_states_label.grid(row=2, column=1, padx=(0,20), pady=(0,5))
        self.sensor_f_discrete_f_states_list = customtkinter.CTkTextbox(self.sensor_f_discrete_f, height=450)
        self.sensor_f_discrete_f_states_list.grid(row=3, column=1, padx=(0,20), pady=(0, 5))
        self.sensor_f_discrete_f_activityName_label = customtkinter.CTkLabel(self.sensor_f_discrete_f, text="...the activity name is")
        self.sensor_f_discrete_f_activityName_label.grid(row=2, column=2, padx=(0,20), pady=(0,5))
        self.sensor_f_discrete_f_activityName_list = customtkinter.CTkTextbox(self.sensor_f_discrete_f, height=450)
        self.sensor_f_discrete_f_activityName_list.grid(row=3, column=2, padx=(0,20), pady=(0, 5))
        self.sensor_f_discrete_f_extract_events_button = customtkinter.CTkButton(self.sensor_f_discrete_f, text='Extract Events', command=self.discrete_extract_events)
        self.sensor_f_discrete_f_extract_events_button.grid(row=2, column=3, padx=(0,5), pady=(0, 5))
        self.sensor_f_discrete_f_extract_events_button_feedback = customtkinter.CTkLabel(self.sensor_f_discrete_f, text="")
        self.sensor_f_discrete_f_extract_events_button_feedback.grid(row=3, column=3, padx=(0,20), pady=(0,5), sticky="n")


        #create the analysis frame
        self.analysis_f = customtkinter.CTkFrame(self.main_frame)
        self.analysis_f.grid(row=0, column=1, sticky='n', padx=10, pady=10)
        self.analysis_f_header = customtkinter.CTkLabel(self.analysis_f, text='Analysis and Visualization', font=customtkinter.CTkFont(size=20, weight="bold"))
        self.analysis_f_header.grid(row=0, column=1, padx=20, pady=(20, 10))
        self.analysis_f_text_descr = customtkinter.CTkLabel(self.analysis_f, text='Create the final OCEL file and show statistics and visualizations.')
        self.analysis_f_text_descr.grid(row=1, column=1, padx=20, pady=(20, 10))
        self.analysis_f_button_createOCEL = customtkinter.CTkButton(self.analysis_f, text='Create final OCEL', command=self.analysis_create_overall_ocel)
        self.analysis_f_button_createOCEL.grid(row=2, column=1, padx=20, pady=(20, 10))
        self.analysis_f_button_createOCEL_feedback = customtkinter.CTkLabel(self.analysis_f, text='')
        self.analysis_f_button_createOCEL_feedback.grid(row=3, column=1, padx=20, pady=(5, 5))
        self.analysis_f_button_analyze = customtkinter.CTkButton(self.analysis_f, text='Analyze/Visualize', command=self.analysis_analyze_overall_ocel)
        #self.analysis_f_button_analyze.grid(row=4, column=1, padx=20, pady=(20, 10))

        #frame in frame (for analysis/visualization)
        self.analysis_f_statistics_f = customtkinter.CTkFrame(self.analysis_f)
        #self.analysis_f_statistics_f.grid(row=5, column=1, sticky='n', padx=10, pady=10) #grid when this frame should be shown.
        self.analysis_f_statistics_f_descr = customtkinter.CTkLabel(self.analysis_f_statistics_f, text='Statistics\n (Visualization saved to session folder)')
        self.analysis_f_statistics_f_descr.grid(row=1, column=1, columnspan=4, padx=20, pady=(20, 10))

        self.analysis_f_statistics_f_numbEvents_label = customtkinter.CTkLabel(self.analysis_f_statistics_f, text='Number of events:')
        self.analysis_f_statistics_f_numbEvents_label.grid(row=2, column=1, padx=(20,5), pady=(20, 10), sticky="e")
        self.analysis_f_statistics_f_numbEvents_value = customtkinter.CTkLabel(self.analysis_f_statistics_f, text='', bg_color="black")
        self.analysis_f_statistics_f_numbEvents_value.grid(row=2, column=2, padx=(0,20), pady=(20, 10), sticky="w")
        self.analysis_f_statistics_f_numbObj_label = customtkinter.CTkLabel(self.analysis_f_statistics_f, text='Number of objects:')
        self.analysis_f_statistics_f_numbObj_label.grid(row=3, column=1, padx=(20,5), pady=(20, 10), sticky="e")
        self.analysis_f_statistics_f_numbObj_value = customtkinter.CTkLabel(self.analysis_f_statistics_f, text='', bg_color="black")
        self.analysis_f_statistics_f_numbObj_value.grid(row=3, column=2, padx=(0,20), pady=(20, 10), sticky="w")
        self.analysis_f_statistics_f_numbEventsVideos_label = customtkinter.CTkLabel(self.analysis_f_statistics_f, text='Number of events from videos:')
        self.analysis_f_statistics_f_numbEventsVideos_label.grid(row=2, column=3, padx=(20,5), pady=(20, 10), sticky="e")
        self.analysis_f_statistics_f_numbEventsVideos_value = customtkinter.CTkLabel(self.analysis_f_statistics_f, text='', bg_color="black")
        self.analysis_f_statistics_f_numbEventsVideos_value.grid(row=2, column=4, padx=(0,20), pady=(20, 10), sticky="w")
        self.analysis_f_statistics_f_numbEventsSensors_label = customtkinter.CTkLabel(self.analysis_f_statistics_f, text='Number of events from sensors:')
        self.analysis_f_statistics_f_numbEventsSensors_label.grid(row=3, column=3, padx=(20,5), pady=(20, 10), sticky="e")
        self.analysis_f_statistics_f_numbEventsSensors_value = customtkinter.CTkLabel(self.analysis_f_statistics_f, text='', bg_color="black")
        self.analysis_f_statistics_f_numbEventsSensors_value.grid(row=3, column=4, padx=(0,20), pady=(20, 10), sticky="w")

        self.analysis_f_statistics_f_ObjTypes_label = customtkinter.CTkLabel(self.analysis_f_statistics_f, text='Object Types')
        self.analysis_f_statistics_f_ObjTypes_label.grid(row=4, column=1, columnspan=2, padx=(5,5), pady=(20, 0))
        self.analysis_f_statistics_f_objTypes_values = customtkinter.CTkTextbox(self.analysis_f_statistics_f, height=450)
        self.analysis_f_statistics_f_objTypes_values.grid(row=5, column=1, columnspan=2, padx=(5,5), pady=(5, 10))

        self.analysis_f_statistics_f_EventTypes_label = customtkinter.CTkLabel(self.analysis_f_statistics_f, text='Event Types / Activities')
        self.analysis_f_statistics_f_EventTypes_label.grid(row=4, column=3, columnspan=2, padx=(5,5), pady=(20, 0))
        self.analysis_f_statistics_f_EventTypes_values = customtkinter.CTkTextbox(self.analysis_f_statistics_f, height=450)
        self.analysis_f_statistics_f_EventTypes_values.grid(row=5, column=3, columnspan=2, padx=(5,5), pady=(5, 10))


        self.show_frame("start")
        self.curr_video_session_id = ""
        
    def show_frame(self, frame_name):
        self.start_f.grid_remove()
        self.process_log_f.grid_remove()
        self.video_labeler_f.grid_remove()
        self.video_obj_area_f.grid_remove()
        self.video_event_f.grid_remove()
        self.sensor_f.grid_remove()
        self.analysis_f.grid_remove()

        if frame_name == "start":
            #self.start_f.tkraise()
            self.start_f.grid(row=0, column=1, sticky='n', padx=10, pady=10)
        elif frame_name == "process_log":
            #self.process_log_f.tkraise()
            self.process_log_f.grid(row=0, column=1, sticky='n', padx=10, pady=10)
        elif frame_name == "video_labeler":
            #self.video_f.tkraise()
            self.video_labeler_f.grid(row=0, column=1, sticky='n', padx=10, pady=10)
        elif frame_name == "video_object_area":
            #self.video_f.tkraise()
            self.video_obj_area_f.grid(row=0, column=1, sticky='n', padx=10, pady=10)
            self.init_object_area_definer_page()
        elif frame_name == "video_event":
            #self.video_f.tkraise()
            self.video_event_f.grid(row=0, column=1, sticky='n', padx=10, pady=10)
            self.init_event_extractor_page()
        elif frame_name == "sensor":
            #self.sensor_f.tkraise()
            self.sensor_f.grid(row=0, column=1, sticky='n', padx=10, pady=10)
        elif frame_name == "analysis":
            #self.analysis_f.tkraise()
            self.analysis_f.grid(row=0, column=1, sticky='n', padx=10, pady=10)
        else:
            print("frame_name not found")

    def enable_page(self, page_name):
        #self.sidebar_button_start.configure(state="disabled")
        #self.sidebar_button_process_log.configure(state="disabled")
        #self.sidebar_button_video.configure(state="disabled")
        #self.sidebar_button_sensor.configure(state="disabled")
        #self.sidebar_button_analysis.configure(state="disabled")
        if page_name == "start":
            self.sidebar_button_start.configure(state="normal")
        elif page_name == "process_log":
            self.sidebar_button_process_log.configure(state="normal")
        elif page_name == "video_labeler":
            self.sidebar_button_video_labeler.configure(state="normal")
        elif page_name == "video_object_area":
            self.sidebar_button_object_area.configure(state="normal")
        elif page_name == "video_event":
            self.sidebar_button_video_event.configure(state="normal")
        elif page_name == "sensor":
            self.sidebar_button_sensor.configure(state="normal")
        elif page_name == "analysis":
            self.sidebar_button_analysis.configure(state="normal")
        else:
            print("page_name not found")

    def disable_page(self, page_name):
        if page_name == "start":
            self.sidebar_button_start.configure(state="disabled")
        elif page_name == "process_log":
            self.sidebar_button_process_log.configure(state="disabled")
        elif page_name == "video_labeler":
            self.sidebar_button_video_labeler.configure(state="disabled")
        elif page_name == "video_object_area":
            self.sidebar_button_object_area.configure(state="disabled")
        elif page_name == "video_event":
            self.sidebar_button_video_event.configure(state="disabled")
        elif page_name == "sensor":
            self.sidebar_button_sensor.configure(state="disabled")
        elif page_name == "analysis":
            self.sidebar_button_analysis.configure(state="disabled")
        elif page_name == "all":
            self.sidebar_button_start.configure(state="disabled")
            self.sidebar_button_process_log.configure(state="disabled")
            self.sidebar_button_video_labeler.configure(state="disabled")
            self.sidebar_button_object_area.configure(state="disabled")
            self.sidebar_button_video_event.configure(state="disabled")
            self.sidebar_button_sensor.configure(state="disabled")
            self.sidebar_button_analysis.configure(state="disabled")
        else:
            print("page_name not found")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def sidebar_button_start_click(self):
        print("sidebar_button_start_click")
        self.show_frame("start")

    def sidebar_button_process_log_click(self):
        print("sidebar_button_process_log_click")
        self.show_frame("process_log")

    def sidebar_button_video_labeler_click(self):
        print("sidebar_button_video_labeler_click")
        self.show_frame("video_labeler")

    def sidebar_button_object_area_click(self):
        print("sidebar_button_object_area_click")
        self.show_frame("video_object_area")

    def sidebar_button_video_event_click(self):
        print("sidebar_button_video_event_click")
        self.show_frame("video_event")

    def sidebar_button_sensor_click(self):
        print("sidebar_button_sensor_click")
        self.show_frame("sensor")

    def sidebar_button_analysis_click(self):
        print("sidebar_button_analysis_click")
        self.show_frame("analysis")

    def start_new_session(self):
        print("start new session")
        path_new_session_folder = filedialog.askdirectory()
        if path_new_session_folder == "":
            print("Folder selection aborted by user. Empty path returned")
            self.disable_page("all")
            self.enable_page("start")
            self.start_feedback_text.configure(text="No folder for new session selected.", text_color="red")
            return
        path_new_session_folder = path_new_session_folder + "/"
        print("Selected path to new session:", path_new_session_folder)
        new_folder_name_path = path_new_session_folder + "session_" + datetime.today().strftime('%Y%m%d_%H%M%S')
        self.session_path = new_folder_name_path + "/"
        if not os.path.exists(new_folder_name_path):
            os.makedirs(new_folder_name_path)
            #os.makedirs(new_folder_name_path + "/labeling_images")
            os.makedirs(new_folder_name_path + "/process_logs_preprocessed")
            os.makedirs(new_folder_name_path + "/results")
            #create session status file
            with open('data/session_status_template.json', "r") as status_json:
                data = json.load(status_json)
            with open(new_folder_name_path + "/status.json", "w") as session_status_file:
                json.dump(data, session_status_file)
        self.start_feedback_text.configure(text="Succesfully created new session folder.\n Proceed to the next step in the menu.", text_color="green")
        self.disable_page("all")
        self.enable_page("process_log")

    def load_session(self):
        print("load session")
        path_existing_session_folder = filedialog.askdirectory()
        if path_existing_session_folder == "":
            print("Folder selection aborted by user. Empty path returned")
            self.disable_page("all")
            self.enable_page("start")
            self.start_feedback_text.configure(text="No session folder selected.", text_color="red")
            return
        else:
            path_existing_session_folder = path_existing_session_folder + "/"
        if os.path.isdir(path_existing_session_folder + 'process_logs_preprocessed'):
            self.session_path = path_existing_session_folder
            print("Selected path to existing session:", path_existing_session_folder)
            self.start_feedback_text.configure(text="Succesfully selected existing session folder.\n Proceed to the next step in the menu.", text_color="green")
            self.disable_page("all")
            #load status file
            with open(path_existing_session_folder + 'status.json', "r") as status_json:
                status_data = json.load(status_json)
            if status_data["final_json_created"] == True:
                print("Last step of the loaded existing session: All steps were finished. Going into analysis.")
                self.enable_page("analysis")
            elif status_data["sensor_events_created"] == True and status_data["sensor_objects_created"] == True:
                print("Last step of the loaded existing session: Sensor events and objects were created.")
                self.enable_page("analysis")
            elif status_data["sensor_events_created"] == True or status_data["sensor_objects_created"] == True:
                print("Last step of the loaded existing session: Sensor events or objects were created but not both.")
                self.enable_page("sensor")
            elif status_data["video_events_created"] == True:
                print("Last step of the loaded existing session: Video events and objects were created.")
                self.enable_page("sensor")
            elif status_data["video_object_areas_defined"] == True:
                print("Last step of the loaded existing session: Video events were created.")
                self.enable_page("video_event")
            elif status_data["video_objects_all_labeled"] == True:
                print("Last step of the loaded existing session: Tracking results in video step were created but video step not finished")
                self.enable_page("video_object_area")
            elif status_data["video_tracking_results_created"] == True:
                print("Last step of the loaded existing session: Tracking results in video step were created but video step not finished")
                self.enable_page("video_labeler")
            elif status_data["process_logs_events_created"] == True and status_data["process_logs_objects_created"] == True:
                print("Last step of the loaded existing session: Process Log events and objects were created")
                self.enable_page("video_labeler")
            elif status_data["process_logs_events_created"] == True or status_data["process_logs_objects_created"] == True:
                print("Last step of the loaded existing session: Process Log events or objects were created but not both")
                self.enable_page("process_log")
            else:
                print("Last step of the loaded existing session: No step was finished so far")
                self.enable_page("process_log")
        else:
            print("Selected path is not a valid session folder. Must contain folder named 'process_logs_preprocessed'.", path_existing_session_folder)
            self.disable_page("all")
            self.enable_page("start")
            self.start_feedback_text.configure(
                text="Selected path is not a valid session folder.\n Session folder must contain folder named 'process_logs_preprocessed'.", text_color="red")

    def set_status(self, key, value):
        if value == True or value == False:
            with open(self.session_path + 'status.json', "r") as status_json:
                data = json.load(status_json)
            if key in data.keys():
                data[key] = value
                with open(self.session_path + 'status.json', "w") as status_json:
                    status_json.write(json.dumps(data))
            else:
                print("Error in setting status. Key not found in status json file.")
        else:
            print("Error in setting status. Value has to be True or False.")

    #process log processing functions
    def process_log_preprocess_file(self):
        print("process log preprocess file")
        file_path = filedialog.askopenfilename()
        if file_path == "":
            print("File selection aborted by user. Empty path returned")
            self.process_log_feedback_text_1.configure(text="No file selected.", text_color="red")
            return
        print("Path to file to be preprocessed:", file_path)
        func_feedback = preprocessProcessLog(file_path, self.session_path)
        if func_feedback == "Success":
            self.process_log_feedback_text_1.configure(text="Succesfully preprocessed and saved file.", text_color="green")
        else:
            self.process_log_feedback_text_1.configure(text=func_feedback, text_color="red")

    def prepr_process_log_extract_events_objects(self):
        print("get events and objects from preprocessed process log")
        file_path = filedialog.askopenfilename()
        if file_path == "":
            print("File selection aborted by user. Empty path returned")
            self.process_log_feedback_text_2.configure(text="No file selected.", text_color="red")
            return
        print("Path to file:", file_path)
        success, ocel_json_data = processLogToOCEL(file_path)
        if success == "Success":
            self.process_log_feedback_text_2.configure(text="Succesfully extracted objects and events from file.", text_color="green")
            #Set the valid ocel file as the main final ocel file that can be enriched in further steps
            with open(self.session_path + 'final_ocel2.json', "w") as final_ocel_file:
                json.dump(ocel_json_data, final_ocel_file)
            self.disable_page("all")
            self.enable_page("video")
            self.set_status("process_logs_objects_created", True)
            self.set_status("process_logs_events_created", True)
        else:
            self.process_log_feedback_text_2.configure(text=success, text_color="red")

    def ocel_file_extract_events_objects(self):
        print("get events and objects from ocel")
        file_path = filedialog.askopenfilename()
        if file_path == "":
            print("File selection aborted by user. Empty path returned")
            self.process_log_feedback_text_3.configure(text="No file selected.", text_color="red")
            return
        print("Path to file:", file_path)
        success, ocel = objects_events_from_ocel(file_path)
        if success == "Success":
            self.process_log_feedback_text_3.configure(text="Succesfully extracted objects and events from ocel file.", text_color="green")
            #Set the valid ocel file as the main final ocel file that can be enriched in further steps
            pm4py.write.write_ocel2_json(ocel, self.session_path + 'final_ocel2.json')
            replace_null_qualifier_ocel2(self.session_path + 'final_ocel2.json')
            self.disable_page("all")
            self.enable_page("video_labeler")
            self.set_status("process_logs_objects_created", True)
            self.set_status("process_logs_events_created", True)
        else:
            self.process_log_feedback_text_3.configure(text=success, text_color="red")

    #video processing functions:
    def run_object_tracking(self):
        print("run_object_tracking")
        self.video_labeler_f_feedback_runObjTracking.configure(text="Running object tracking. This step may take some time...", text_color="green")
        self.video_labeler_f_feedback_loadObjTracking.configure(text="")
        path_video_file = filedialog.askopenfilename()
        if path_video_file == "":
            print("Folder selection aborted by user. Empty path returned")
            self.video_labeler_f_feedback_runObjTracking.configure(text="No file selected.", text_color="red")
            return
        if (not path_video_file.endswith(".MOV")) and (not path_video_file.endswith(".mp4")):
            print("Video file must be .MOV or .mp4")
            self.video_labeler_f_feedback_runObjTracking.configure(text="Video file must be .MOV or .mp4", text_color="red")
            return
        #print("path video file:", path_video_file)
        self.curr_video_session_id = Path(path_video_file).stem
        path_new_video_dir = self.session_path + self.curr_video_session_id
        if not os.path.exists(path_new_video_dir):
            os.makedirs(path_new_video_dir)
            os.makedirs(path_new_video_dir + "/labeling_images")

        stream, model, tracker, device = init_object_tracking(path_video_file)
        tracking_results = object_tracking(stream, model, tracker, device, confidence=0.3)
        self.path_curr_tracking_results = self.session_path + self.curr_video_session_id + "/tracking_results.pkl"
        tracking_results.to_pickle(self.path_curr_tracking_results) #save tracking results in folder of current session
        #create file for video meta data (e.g., fps)
        curr_fps = stream.get(cv2.CAP_PROP_FPS)
        data = {}
        data['video_fps'] = curr_fps
        with open(path_new_video_dir + "/metadata.json", "w") as video_metadata:
            json.dump(data, video_metadata)
        #print(tracking_results)
        #tracking_results = pd.read_pickle(self.path_curr_tracking_results)
        #tracking_results.to_excel(self.session_path + self.curr_video_session_id + "/tracking_results.xlsx")
        object_labeling_image_prep(stream, tracking_results, self.session_path, self.curr_video_session_id, sample_size=2)
        manual_object_definer_prep(stream, self.session_path, self.curr_video_session_id)
        self.video_labeler_f_feedback_runObjTracking.configure(text="Successfully ran object tracking and saved results", text_color="green")
        self.set_status("video_tracking_results_created", True)
        self.initialize_labeling()

    def load_object_tracking(self):
        print("load_object_tracking")
        self.video_labeler_f_feedback_runObjTracking.configure(text="")

        path_obj_tracking_results = filedialog.askopenfilename()
        if path_obj_tracking_results == "":
            print("Folder selection aborted by user. Empty path returned")
            self.video_labeler_f_feedback_loadObjTracking.configure(text="No file selected.", text_color="red")
            return
        if not path_obj_tracking_results.endswith(".pkl"):
            print("Object tracking results file must be .pkl")
            self.video_labeler_f_feedback_loadObjTracking.configure(text="Object tracking results file must be .pkl", text_color="red")
            return
        self.path_curr_tracking_results = path_obj_tracking_results
        self.curr_video_session_id = os.path.basename(os.path.dirname(path_obj_tracking_results))
        self.video_labeler_f_feedback_loadObjTracking.configure(text="Succesfully loaded 'tracking_results.pkl' file.", text_color="green")
        self.initialize_labeling()

    def initialize_labeling(self):
        print("initialize_labeling")
        #get maximum ID from object tracking df
        tracking_results_df = pd.read_pickle(self.path_curr_tracking_results)
        self.max_id_obj_tracking = len(tracking_results_df["object_id"].unique())
        self.all_objIDs_list = list(tracking_results_df["object_id"].unique())
        self.video_labeler_labeling_f.grid(row=4, column=1, columnspan=18, sticky='n', padx=10, pady=10) #show the labeling frame
        self.video_labeling_curr_id = self.all_objIDs_list[0]
        self.vid_labeling_load_img(self.video_labeling_curr_id) #load first image (with id 1)
        self.video_labeling_load_dropdowns()
        self.video_labeling_set_data()
        self.set_ignore_switch_state()

    def video_labeling_load_dropdowns(self):
        print("video_labeling_load_dropdowns")
        all_obj_ids_list = get_all_object_ids(self.session_path)
        self.video_labelling_f_objID_dropd.configure(values=all_obj_ids_list)
        all_obj_types_list = get_all_object_types(self.session_path)
        self.video_labelling_f_objType_dropd.configure(values=all_obj_types_list)

    def vid_labeling_get_next_obj_id(self):
        if self.video_labeling_curr_id == self.all_objIDs_list[-1]:
            return self.all_objIDs_list[0]
        else:
            new_index = self.all_objIDs_list.index(self.video_labeling_curr_id) + 1
            return self.all_objIDs_list[new_index]
        
    def vid_labeling_get_previous_obj_id(self):
        if self.video_labeling_curr_id == self.all_objIDs_list[0]:
            return self.all_objIDs_list[-1]
        else:
            new_index = self.all_objIDs_list.index(self.video_labeling_curr_id) - 1
            return self.all_objIDs_list[new_index]
    
    def vid_labeling_load_img(self,object_id):
        img_path = self.session_path + self.curr_video_session_id + '/labeling_images/' + str(object_id) + '.jpg'
        img = customtkinter.CTkImage(light_image=Image.open(img_path), dark_image=Image.open(img_path),
                                     size=(1152,648)) #WidthxHeight #old size: 1152,648
        self.video_labelling_f_image = customtkinter.CTkLabel(self.video_labeler_labeling_f, image = img)
        self.video_labelling_f_image.grid(row=2, rowspan=48, column=1, columnspan=7, padx=20, pady=(20, 10))
        self.video_labelling_f_curr_obj_id_number_label.configure(text=str(self.all_objIDs_list.index(object_id)+1) + '/' + str(self.max_id_obj_tracking) + ' Objects') #display new id

    def get_obj_attributes_list(self):
        attr_list = []
        textBox_names = self.video_labelling_f_objAttr_names_list.get(0.0, "end")
        allNames = textBox_names.split(sep="\n")
        textBox_values = self.video_labelling_f_objAttr_values_list.get(0.0, "end")
        allValues = textBox_values.split(sep="\n")
        for index, attr_name in enumerate(allNames):
            if not attr_name == "":
                curr_attribute = {
                    "name":attr_name,
                    "value":allValues[index],
                    "time":""
                }
                attr_list.append(curr_attribute)
        return attr_list

    def ignore_object(self):
        ignore_obj_val = self.video_labelling_f_button_ignoreObj.get() #is 0 or 1
        tracking_results = pd.read_pickle(self.path_curr_tracking_results) #load results of current session
        if ignore_obj_val == 1:
            tracking_results.loc[tracking_results.object_id == self.video_labeling_curr_id, ['ignore_object']] = True
        else:
            tracking_results.loc[tracking_results.object_id == self.video_labeling_curr_id, ['ignore_object']] = False
        tracking_results.to_pickle(self.path_curr_tracking_results) #save and overwrite updated results in folder of current session
        #print(tracking_results)

    def set_ignore_switch_state(self):
        tracking_results = pd.read_pickle(self.path_curr_tracking_results)
        if tracking_results[tracking_results.object_id == self.video_labeling_curr_id].iloc[0, tracking_results.columns.get_loc("ignore_object")]:
            self.video_labelling_f_button_ignoreObj.select() #turn switch on/value=1
        else:
            self.video_labelling_f_button_ignoreObj.deselect() #turn switch off/value=0

    def next_object(self):
        #save/update data
        curr_obj_id = self.video_labelling_f_objID_dropd.get()
        curr_obj_type = self.video_labelling_f_objType_dropd.get()
        curr_attr_list = self.get_obj_attributes_list()
        update_or_create_object(self.session_path, curr_obj_id, curr_obj_type, curr_attr_list)
        #update also in the tracking results for later use in event extractor
        object_labeling_post_annotator(self.path_curr_tracking_results, self.video_labeling_curr_id, curr_obj_id, curr_obj_type, curr_attr_list)
        #load next image
        next_obj_id = self.vid_labeling_get_next_obj_id()
        self.vid_labeling_load_img(next_obj_id)
        self.video_labeling_load_dropdowns()
        self.video_labeling_curr_id = next_obj_id
        self.video_labeling_set_data()
        #set ignore switch state
        self.set_ignore_switch_state()

    def previous_object(self):
        #save/update data
        curr_obj_id = self.video_labelling_f_objID_dropd.get()
        curr_obj_type = self.video_labelling_f_objType_dropd.get()
        curr_attr_list = self.get_obj_attributes_list()
        update_or_create_object(self.session_path, curr_obj_id, curr_obj_type, curr_attr_list)
        #update also in the tracking results for later use in event extractor
        object_labeling_post_annotator(self.path_curr_tracking_results, self.video_labeling_curr_id, curr_obj_id, curr_obj_type, curr_attr_list)
        #load previous image
        previous_obj_id = self.vid_labeling_get_previous_obj_id()
        self.vid_labeling_load_img(previous_obj_id)
        self.video_labeling_load_dropdowns()
        self.video_labeling_curr_id = previous_obj_id
        self.video_labeling_set_data()
        #set ignore switch state
        self.set_ignore_switch_state()

    def video_labeling_set_data(self):
        print("video_labeling_set_data")
        #check the already updated tracking_results.pkl if the current object (by old tracking ID) was already assigned new data to
        tracking_results_df = pd.read_pickle(self.path_curr_tracking_results)
        curr_obj_only_df = tracking_results_df.loc[tracking_results_df['object_id'] == self.video_labeling_curr_id]
        if not curr_obj_only_df["object_id_manual"].iloc[0] == "":
            #If the data was already set manually before then display it
            curr_obj_id = str(curr_obj_only_df["object_id_manual"].iloc[0])
            self.video_labelling_f_objID_dropd.configure(variable=customtkinter.StringVar(value=curr_obj_id))
            object_type, object_attributes = get_object_type_and_attributes(self.session_path, curr_obj_id)
            self.video_labelling_f_objType_dropd.configure(variable=customtkinter.StringVar(value=object_type))
            self.video_labelling_f_objAttr_names_list = customtkinter.CTkTextbox(self.video_labeler_labeling_f)
            self.video_labelling_f_objAttr_values_list = customtkinter.CTkTextbox(self.video_labeler_labeling_f)
            for attribute in object_attributes:
                self.video_labelling_f_objAttr_names_list.insert("end", attribute["name"] + "\n") #index is 'line.character'
                self.video_labelling_f_objAttr_values_list.insert("end", attribute["value"] + "\n") #index is 'line.character'
            self.video_labelling_f_objAttr_names_list.grid(row=10, column=8, padx=1, pady=(0, 10))
            self.video_labelling_f_objAttr_values_list.grid(row=10, column=9, padx=0, pady=(0, 10))
        else:
            #If data for the object was not set manually before then display empty data fields
            self.video_labelling_f_objID_dropd.configure(variable=customtkinter.StringVar(value=""))
            self.video_labelling_f_objType_dropd.configure(variable=customtkinter.StringVar(value=""))
            self.video_labelling_f_objAttr_names_list = customtkinter.CTkTextbox(self.video_labeler_labeling_f)
            self.video_labelling_f_objAttr_values_list = customtkinter.CTkTextbox(self.video_labeler_labeling_f)
            self.video_labelling_f_objAttr_names_list.grid(row=10, column=8, padx=1, pady=(0, 10))
            self.video_labelling_f_objAttr_values_list.grid(row=10, column=9, padx=0, pady=(0, 10))

    def finish_labeling(self):
        print("finish_labeling")
        #save/update data from the current object
        curr_obj_id = self.video_labelling_f_objID_dropd.get()
        curr_obj_type = self.video_labelling_f_objType_dropd.get()
        curr_attr_list = self.get_obj_attributes_list()
        update_or_create_object(self.session_path, curr_obj_id, curr_obj_type, curr_attr_list)
        #update also in the tracking results for later use in event extractor
        object_labeling_post_annotator(self.path_curr_tracking_results, self.video_labeling_curr_id, curr_obj_id, curr_obj_type, curr_attr_list)
        #delete all ignored objects from the tracking results list (delete all rows with this old object id)
        self.set_status("video_objects_all_labeled", True)
        self.disable_page("all")
        self.enable_page("video_object_area")
        self.show_frame("video_object_area")
    
    def dropdwon_objID_clicked(self, choice):
        print("combobox dropdown clicked:", choice)
        #display the corresponding object type and attributes of the selected object id
        object_type, object_attributes = get_object_type_and_attributes(self.session_path, choice)
        self.video_labelling_f_objType_dropd.configure(variable=customtkinter.StringVar(value=object_type))

        self.video_labelling_f_objAttr_names_list = customtkinter.CTkTextbox(self.video_labeler_labeling_f)
        self.video_labelling_f_objAttr_values_list = customtkinter.CTkTextbox(self.video_labeler_labeling_f)
        for attribute in object_attributes:
            self.video_labelling_f_objAttr_names_list.insert("end", attribute["name"] + "\n") #index is 'line.character'
            self.video_labelling_f_objAttr_values_list.insert("end", attribute["value"] + "\n") #index is 'line.character'
        self.video_labelling_f_objAttr_names_list.grid(row=10, column=8, padx=1, pady=(0, 10))
        self.video_labelling_f_objAttr_values_list.grid(row=10, column=9, padx=0, pady=(0, 10))


    #Video Object Area Definer Functions
    def init_object_area_definer_page(self):
        if self.curr_video_session_id == "":
            #application was started recently
            self.video_obj_area_f_sel_trackResults_feedback.configure(text="Please select a tracking_results.pkl file from the video session you like.", text_color="red")
        else:
            #application has been running and tracking results had been selected before
            self.video_obj_area_f_sel_trackResults_feedback.configure(text="Automatically loaded 'tracking_results.pkl' file from session: " + self.curr_video_session_id, text_color="green")
            #enable/grid the frame in frame for actual area definer
            self.video_obj_area_f_definer.grid(row=4, column=1, columnspan=18, sticky='n', padx=10, pady=10)
            sample_img_path = self.session_path + self.curr_video_session_id + '/object_definition_sample_img.jpg'
            img = customtkinter.CTkImage(light_image=Image.open(sample_img_path), dark_image=Image.open(sample_img_path), size=(576,648)) #WidthxHeight
            self.video_obj_area_f_image = customtkinter.CTkLabel(self.video_obj_area_f_definer, image = img)
            self.video_obj_area_f_image.grid(row=3, rowspan=47, column=1, columnspan=7, padx=20, pady=(20, 10))
            all_obj_types_list = get_all_object_types(self.session_path)
            self.video_obj_area_f_objType_dropd.configure(values=all_obj_types_list)
            self.bounding_box = [0, 0, 0, 0]

    def area_definer_select_tracking_results(self):
        path_obj_tracking_results = filedialog.askopenfilename()
        if path_obj_tracking_results == "":
            print("Folder selection aborted by user. Empty path returned")
            self.video_obj_area_f_sel_trackResults_feedback.configure(text="No file selected.", text_color="red")
            return
        if not path_obj_tracking_results.endswith(".pkl"):
            print("Object tracking results file must be .pkl")
            self.video_obj_area_f_sel_trackResults_feedback.configure(text="Object tracking results file must be .pkl", text_color="red")
            return
        self.path_curr_tracking_results = path_obj_tracking_results
        self.curr_video_session_id = os.path.basename(os.path.dirname(path_obj_tracking_results))
        self.video_obj_area_f_sel_trackResults_feedback.configure(text="Succesfully loaded 'tracking_results.pkl' file from session: " + self.curr_video_session_id, text_color="green")
        self.video_obj_area_f_definer.grid(row=4, column=1, columnspan=18, sticky='n', padx=10, pady=10)
        sample_img_path = self.session_path + self.curr_video_session_id + '/object_definition_sample_img.jpg'
        img = customtkinter.CTkImage(light_image=Image.open(sample_img_path), dark_image=Image.open(sample_img_path), size=(576,648)) #WidthxHeight
        self.video_obj_area_f_image = customtkinter.CTkLabel(self.video_obj_area_f_definer, image = img)
        self.video_obj_area_f_image.grid(row=3, rowspan=47, column=1, columnspan=7, padx=20, pady=(20, 10))
        #Reset Object ID, Type, Attributes, and Feedback Texts
        self.video_obj_area_f_objID_dropd.delete("0.0", "end")
        self.video_obj_area_f_objType_dropd.configure(variable=customtkinter.StringVar(value=""))
        all_obj_types_list = get_all_object_types(self.session_path)
        self.video_obj_area_f_objType_dropd.configure(values=all_obj_types_list)
        self.video_obj_area_f_objAttr_names_list.delete("0.0", "end")
        self.video_obj_area_f_objAttr_values_list.delete("0.0", "end")
        self.video_obj_area_f_button_select_bb_feedback.configure(text="", text_color="green")
        self.video_obj_area_f_button_save_feedback.configure(text="", text_color="green")
        self.bounding_box = [0, 0, 0, 0]

    def select_bounding_box(self):
        bb_normal, bb_normalized = select_bounding_boxes(self.session_path, self.curr_video_session_id)
        self.bounding_box = bb_normalized
        if self.bounding_box == [0, 0, 0, 0]:
            print("bounding_box in yolo format (xywh with center x y and normalized)", self.bounding_box)
            self.video_obj_area_f_button_select_bb_feedback.configure(text="No Bounding Box selected.", text_color="red")
            #display sample image again
            img_path = self.session_path + self.curr_video_session_id + '/object_definition_sample_img.jpg'
            img = customtkinter.CTkImage(light_image=Image.open(img_path), dark_image=Image.open(img_path), size=(576,648)) #WidthxHeight
            self.video_obj_area_f_image = customtkinter.CTkLabel(self.video_obj_area_f_definer, image = img)
            self.video_obj_area_f_image.grid(row=3, rowspan=47, column=1, columnspan=7, padx=20, pady=(20, 10))
        else:
            print("bounding_box in yolo format (xywh with center x y and normalized)", self.bounding_box)
            left, top, right, bottom = create_image_with_bounding_box(self.session_path, self.curr_video_session_id, bb_normal)
            img_path = self.session_path + self.curr_video_session_id + '/curr_object_area.jpg'
            img = customtkinter.CTkImage(light_image=Image.open(img_path), dark_image=Image.open(img_path), size=(576,648)) #WidthxHeight
            self.video_obj_area_f_image = customtkinter.CTkLabel(self.video_obj_area_f_definer, image = img)
            self.video_obj_area_f_image.grid(row=3, rowspan=47, column=1, columnspan=7, padx=20, pady=(20, 10))
            self.video_obj_area_f_button_select_bb_feedback.configure(text=f"Succesfully selected Bounding Box (left, top, right, bottom): {left}, {top}, {right}, {bottom}",
                                                                      text_color="green")
            self.video_obj_area_f_button_save_feedback.configure(text="", text_color="green")

    def get_obj_attributes_list_area_definer(self):
        attr_list = []
        textBox_names = self.video_obj_area_f_objAttr_names_list.get(0.0, "end")
        allNames = textBox_names.split(sep="\n")
        textBox_values = self.video_obj_area_f_objAttr_values_list.get(0.0, "end")
        allValues = textBox_values.split(sep="\n")
        for index, attr_name in enumerate(allNames):
            if not attr_name == "":
                curr_attribute = {
                    "name":attr_name,
                    "value":allValues[index],
                    "time":""
                }
                attr_list.append(curr_attribute)
        return attr_list

    def save_defined_object(self):
        print("save_defined_object")
        if self.bounding_box == [0, 0, 0, 0]:
            self.video_obj_area_f_button_save_feedback.configure(text="No bounding box selected.", text_color="red")
            return
        obj_id_value_raw = self.video_obj_area_f_objID_dropd.get(1.0, "end")
        curr_obj_id = obj_id_value_raw.split(sep="\n")[0]
        curr_obj_type = self.video_obj_area_f_objType_dropd.get()
        if curr_obj_id == "":
            self.video_obj_area_f_button_save_feedback.configure(text="No object ID given.", text_color="red")
            return
        if curr_obj_type == "":
            self.video_obj_area_f_button_save_feedback.configure(text="No object type given.", text_color="red")
            return
        all_obj_ids_list = get_all_object_ids(self.session_path)
        if curr_obj_id in all_obj_ids_list:
            self.video_obj_area_f_button_save_feedback.configure(text="Object ID already exists.", text_color="red")
            return
        #Save newly defined object in tracking_results.pkl and in final_ocel2.json
        curr_attr_list = self.get_obj_attributes_list_area_definer()
        defined_object = [curr_obj_id, curr_obj_type, self.bounding_box, curr_attr_list]
        manual_object_definer(self.session_path + self.curr_video_session_id + '/', defined_object)
        update_or_create_object(self.session_path, curr_obj_id, curr_obj_type, curr_attr_list)
        self.video_obj_area_f_button_save_feedback.configure(text="Succesfully saved defined object. Continue with other objects or finish object definer step.",
                                                             text_color="green")

        #Reset the object type dropdown to include all new object types
        self.video_obj_area_f_objType_dropd.configure(variable=customtkinter.StringVar(value=""))
        all_obj_types_list = get_all_object_types(self.session_path)
        self.video_obj_area_f_objType_dropd.configure(values=all_obj_types_list)
        #Reset image
        sample_img_path = self.session_path + self.curr_video_session_id + '/object_definition_sample_img.jpg'
        img = customtkinter.CTkImage(light_image=Image.open(sample_img_path), dark_image=Image.open(sample_img_path), size=(576,648)) #WidthxHeight
        self.video_obj_area_f_image = customtkinter.CTkLabel(self.video_obj_area_f_definer, image = img)
        self.video_obj_area_f_image.grid(row=3, rowspan=47, column=1, columnspan=7, padx=20, pady=(20, 10))
        #Reset the rest
        self.video_obj_area_f_objID_dropd.delete("0.0", "end")
        self.video_obj_area_f_objAttr_names_list.delete("0.0", "end")
        self.video_obj_area_f_objAttr_values_list.delete("0.0", "end")
        self.video_obj_area_f_button_select_bb_feedback.configure(text="", text_color="green")

    def dropdwon_objType_clicked(self, choice):
        print("dropdwon_objType_clicked", choice)

    def finish_object_definer(self):
        print("finish_object_definer")
        self.set_status("video_object_areas_defined", True)
        self.disable_page("all")
        self.enable_page("video_event")
        self.show_frame("video_event")

    #Video Event Extractor Functions
    def init_event_extractor_page(self):
        if self.curr_video_session_id == "":
            #application was started recently
            self.video_event_f_sel_trackResults_feedback.configure(text="Please select a tracking_results.pkl file from the video session you like.", text_color="red")
        else:
            #application has been running and tracking results had been selected before
            self.video_event_f_sel_trackResults_feedback.configure(text="Automatically loaded 'tracking_results.pkl' file from session: " + self.curr_video_session_id, text_color="green")
            #enable/grid the frame in frame for actual area definer
            self.video_event_f_extractor.grid(row=4, column=1, columnspan=18, sticky='n', padx=10, pady=10)
            all_obj_ids_list = get_all_object_ids(self.session_path)
            self.video_event_f_extractor_objID1_dropd.configure(values=all_obj_ids_list)
            self.video_event_f_extractor_objID2_dropd.configure(values=all_obj_ids_list)

    def event_extractor_select_tracking_results(self):
        path_obj_tracking_results = filedialog.askopenfilename()
        if path_obj_tracking_results == "":
            print("Folder selection aborted by user. Empty path returned")
            self.video_event_f_sel_trackResults_feedback.configure(text="No file selected.", text_color="red")
            return
        if not path_obj_tracking_results.endswith(".pkl"):
            print("Object tracking results file must be .pkl")
            self.video_event_f_sel_trackResults_feedback.configure(text="Object tracking results file must be .pkl", text_color="red")
            return
        self.path_curr_tracking_results = path_obj_tracking_results
        self.curr_video_session_id = os.path.basename(os.path.dirname(path_obj_tracking_results))
        self.video_event_f_sel_trackResults_feedback.configure(text="Succesfully loaded 'tracking_results.pkl' file from session: " + self.curr_video_session_id, text_color="green")
        self.video_event_f_extractor.grid(row=4, column=1, columnspan=18, sticky='n', padx=10, pady=10)
        all_obj_ids_list = get_all_object_ids(self.session_path)
        self.video_event_f_extractor_objID1_dropd.configure(values=all_obj_ids_list)
        self.video_event_f_extractor_objID2_dropd.configure(values=all_obj_ids_list)
        #Reset timestamp field, feedbackExtractEvents, objID 1 and 2 dropdowns, eventType text, all 3 lists
        self.video_event_f_extractor_timestamp_input.delete("0.0", "end") #textbox
        self.video_event_f_extractor_button_extractEvents_feedback.configure(text="", text_color="green") #label
        self.video_event_f_extractor_objID1_dropd.configure(variable=customtkinter.StringVar(value=""))
        self.video_event_f_extractor_objID2_dropd.configure(variable=customtkinter.StringVar(value=""))
        self.video_event_f_extractor_eventType_input.delete("0.0", "end")
        self.video_event_f_extractor_objID1_list.delete("0.0", "end")
        self.video_event_f_extractor_objID2_list.delete("0.0", "end")
        self.video_event_f_extractor_eventType_list.delete("0.0", "end")

    def validate_timestamp(self, date_text):
        try:
            if date_text != datetime.strptime(date_text, "%Y-%m-%dT%H:%M:%S.%f").strftime('%Y-%m-%dT%H:%M:%S.%f'):
                raise ValueError
            return True
        except ValueError:
            return False

    def get_rules_list(self):
        rules_list = []
        textBox_objID1 = self.video_event_f_extractor_objID1_list.get(0.0, "end")
        allObjID1s = textBox_objID1.split(sep="\n")
        textBox_objID2 = self.video_event_f_extractor_objID2_list.get(0.0, "end")
        allObjID2s = textBox_objID2.split(sep="\n")
        textBox_eventTypes = self.video_event_f_extractor_eventType_list.get(0.0, "end")
        allEventTypes = textBox_eventTypes.split(sep="\n")
        for index, objID1 in enumerate(allObjID1s):
            if not (objID1 == "" and allObjID2s[index] == "" and allEventTypes[index] == ""):
                curr_rule = [objID1, allObjID2s[index], allEventTypes[index]]
                rules_list.append(curr_rule)
        return rules_list

    def extract_events(self):
        print("extract_events")
        #Check timestamp format and not empty. "1970-01-01T00:00:00.000000"
        timestamp_value_raw = self.video_event_f_extractor_timestamp_input.get(1.0, "end")
        timestamp_value = timestamp_value_raw.split(sep="\n")[0]
        if self.validate_timestamp(timestamp_value):
            timestamp_as_datetime = datetime.strptime(timestamp_value, "%Y-%m-%dT%H:%M:%S.%f")
        else:
            self.video_event_f_extractor_button_extractEvents_feedback.configure(text="Timestamp is not in the valid format.", text_color="red")
            return
        rules_list_of_lists = self.get_rules_list()
        #print(rules_list_of_lists)
        #print(rules_list_of_lists[0])
        all_obj_ids_list = get_all_object_ids(self.session_path)
        #Check if for every entry in objID1 there also exists a value for objID2 and eventType (both not "")
        for index, rule in enumerate(rules_list_of_lists):
            if rule[0] == "" or rule[1] == "" or rule[2] == "":
                feedback_text = f"There need to be values given for Object ID 1, Object ID 2, and Activity Name. This is not the case for rule number {index+1}"
                print(feedback_text)
                self.video_event_f_extractor_button_extractEvents_feedback.configure(text=feedback_text, text_color="red")
                return
            #Check if objID1 and objID2 are valid known object IDs (from final_json2.ocel)
            if not rule[0] in all_obj_ids_list:
                feedback_text = f"Object ID 1 in rule {index+1} is not a known object."
                print(feedback_text)
                self.video_event_f_extractor_button_extractEvents_feedback.configure(text=feedback_text, text_color="red")
                return
            if not rule[1] in all_obj_ids_list:
                feedback_text = f"Object ID 2 in rule {index+1} is not a known object."
                print(feedback_text)
                self.video_event_f_extractor_button_extractEvents_feedback.configure(text=feedback_text, text_color="red")
                return
        #Check if every objID1 and objID2 combo exists only once (also backwards)
        all_rule_combos = []
        for index, rule in enumerate(rules_list_of_lists):
            rule_combo_normal = str(rule[0]) + " " + str(rule[1])
            rule_combo_backwards = str(rule[1]) + " " + str(rule[0])
            if rule_combo_normal in all_rule_combos or rule_combo_backwards in all_rule_combos:
                feedback_text = f"There already exists a rule including these two objects: {rule_combo_normal}."
                print(feedback_text)
                self.video_event_f_extractor_button_extractEvents_feedback.configure(text=feedback_text, text_color="red")
                return
            else:
                all_rule_combos.append(rule_combo_normal)
                all_rule_combos.append(rule_combo_backwards)

        #Create events
        self.video_event_f_extractor_button_extractEvents_feedback.configure(text="Creating events. This may take some time...", text_color="green")
        with open(self.session_path + self.curr_video_session_id + '/metadata.json', "r") as video_metadata:
            video_data = json.load(video_metadata)
        curr_fps = video_data["video_fps"]
        events = video_event_trigger_algorithm_standard(self.session_path, self.curr_video_session_id, timestamp_value, curr_fps, rules_list_of_lists)
        add_events(self.session_path, events) # add events to final_ocel2.json

        self.video_event_f_extractor_button_extractEvents_feedback.configure(text="Succesfully extracted events. Finish step or load another video.", text_color="green")

    def add_rule(self):
        print("add_rule")
        curr_objID1_value = self.video_event_f_extractor_objID1_dropd.get()
        curr_objID2_value = self.video_event_f_extractor_objID2_dropd.get()
        eventType_value_raw = self.video_event_f_extractor_eventType_input.get(1.0, "end")
        curr_eventType_value = eventType_value_raw.split(sep="\n")[0]
        if curr_objID1_value != "" and curr_objID2_value != "" and curr_eventType_value != "":
            self.video_event_f_extractor_objID1_list.insert("end", curr_objID1_value + "\n")
            self.video_event_f_extractor_objID2_list.insert("end", curr_objID2_value + "\n")
            self.video_event_f_extractor_eventType_list.insert("end", curr_eventType_value + "\n")
            self.video_event_f_extractor_objID1_dropd.configure(variable=customtkinter.StringVar(value=""))
            self.video_event_f_extractor_objID2_dropd.configure(variable=customtkinter.StringVar(value=""))
            self.video_event_f_extractor_eventType_input.delete("0.0", "end")
        else:
            print("Object ID 1 or 2 or Activity Name was empty")

    def finish_extract_events(self):
        print("finish_extract_events")
        self.set_status("video_events_created", True)
        self.disable_page("all")
        self.enable_page("sensor")
        self.show_frame("sensor")

    def load_additional_video(self):
        print("load_additional_video")
        self.set_status("video_tracking_results_created", False)
        self.set_status("video_objects_all_labeled", False)
        self.set_status("video_object_areas_defined", False)
        self.set_status("video_events_created", False)
        self.disable_page("all")
        self.enable_page("video_labeler")
        self.show_frame("video_labeler")
    

    #Sensor Functions
    def load_sensor_file(self):
        path_sensor_file = filedialog.askopenfilename()
        if path_sensor_file == "":
            print("Folder selection aborted by user. Empty path returned")
            self.sensor_f_button_load_sensor_file_feedback.configure(text="No file selected.", text_color="red")
            return
        if not path_sensor_file.endswith(".csv"):
            print("Sensor file must be .csv")
            self.sensor_f_button_load_sensor_file_feedback.configure(text="Sensor file must be .csv", text_color="red")
            return
        self.path_curr_sensor_file = path_sensor_file
        self.df_sensor_file = pd.read_csv(path_sensor_file, sep=",")
        self.sensor_f_details_f.grid(row=4, column=1, columnspan=8, sticky='n', padx=10, pady=10)
        self.sensor_f_discrete_f.grid_remove()
        self.sensor_f_continuous_f.grid(row=5, column=1, columnspan=8, sticky='n', padx=10, pady=10)
        all_obj_ids_list = get_all_object_ids(self.session_path)
        self.sensor_f_details_f_objID_dropd.configure(values=all_obj_ids_list)
        all_obj_types_list = get_all_object_types(self.session_path)
        self.sensor_f_details_f_objType_dropd.configure(values=all_obj_types_list)
        self.sensor_f_details_f_relObj_dropd.configure(values=all_obj_ids_list)
        self.curr_sensor_file_name = Path(path_sensor_file).stem
        #empty all inputs: timestamp column, sensor values column, objID, objType, Attribute Names and Values lists, related objects list, continuous data text fields
        self.sensor_f_details_f_time_textbox.delete("0.0", "end") #textbox
        self.sensor_f_details_f_value_textbox.delete("0.0", "end")
        self.sensor_f_details_f_objID_dropd.configure(variable=customtkinter.StringVar(value="")) #dropdown
        self.sensor_f_details_f_objType_dropd.configure(variable=customtkinter.StringVar(value=""))
        self.sensor_f_details_f_attr_name_list.delete("0.0", "end")
        self.sensor_f_details_f_attr_value_list.delete("0.0", "end")
        self.sensor_f_details_f_relObj_dropd.configure(variable=customtkinter.StringVar(value=""))
        self.sensor_f_details_f_relObj_list.delete("0.0", "end")
        self.sensor_f_continuous_f_lastXEntries_greater_text.delete("0.0", "end")
        self.sensor_f_continuous_f_lastXEntries_smaller_text.delete("0.0", "end")
        self.sensor_f_continuous_f_threshold_greater_text.delete("0.0", "end")
        self.sensor_f_continuous_f_threshold_smaller_text.delete("0.0", "end")
        self.sensor_f_continuous_f_activityName_greater_text.delete("0.0", "end")
        self.sensor_f_continuous_f_activityName_smaller_text.delete("0.0", "end")
        self.sensor_f_details_f_time_label_feedback.configure(text="", text_color="green")
        self.sensor_f_discrete_f_states_list.delete("0.0", "end")
        self.sensor_f_discrete_f_activityName_list.delete("0.0", "end")
        self.sensor_f_details_f_discrCont_switch.deselect() #Set switch back to off/0
        self.sensor_f_discrete_f_extract_events_button_feedback.configure(text="", text_color="green")
        self.sensor_f_continuous_f_extract_events_button_feedback.configure(text="", text_color="green")

        self.sensor_f_button_load_sensor_file_feedback.configure(text=f"Succesfully loaded sensor file {self.curr_sensor_file_name}", text_color="green")

    def discr_cont_switch(self):
        discr_cont_switch_val = self.sensor_f_details_f_discrCont_switch.get() #is 0 or 1
        if discr_cont_switch_val == 1:
            #print("discr_cont_switch is On")
            timeColumn_value_raw = self.sensor_f_details_f_time_textbox.get(1.0, "end")
            timeColumn_value = timeColumn_value_raw.split(sep="\n")[0]
            valueColumn_value_raw = self.sensor_f_details_f_value_textbox.get(1.0, "end")
            valueColumn_value = valueColumn_value_raw.split(sep="\n")[0]
            if timeColumn_value == "":
                self.sensor_f_details_f_time_label_feedback.configure(text="Please first input timestamp column.", text_color="red")
                self.sensor_f_details_f_discrCont_switch.deselect() #Set switch back to off/0
                return
            elif valueColumn_value == "":
                self.sensor_f_details_f_time_label_feedback.configure(text="Please first input sensor values column.", text_color="red")
                self.sensor_f_details_f_discrCont_switch.deselect() #Set switch back to off/0
                return
            else:
                if valueColumn_value in self.df_sensor_file.columns.values:
                    sensor_states = self.df_sensor_file[valueColumn_value].unique()
                    #print("sensor_states: ", sensor_states)
                    self.sensor_f_discrete_f_states_list.delete("0.0", "end")
                    #self.sensor_f_discrete_f_activityName_list.delete("0.0", "end")
                    for sensor_state in sensor_states:
                        self.sensor_f_discrete_f_states_list.insert("end", str(sensor_state) + "\n")
                    self.sensor_f_details_f_time_label_feedback.configure(text="", text_color="green")
                    self.sensor_f_continuous_f.grid_remove()
                    self.sensor_f_discrete_f.grid(row=5, column=1, columnspan=8, sticky='n', padx=10, pady=10)
                else:
                    self.sensor_f_details_f_time_label_feedback.configure(text="Sensor values column not found in .csv file.", text_color="red")
                    self.sensor_f_details_f_discrCont_switch.deselect() #Set switch back to off/0
                    return
        else:
            #print("discr_cont_switch is Off")
            self.sensor_f_discrete_f.grid_remove()
            self.sensor_f_continuous_f.grid(row=5, column=1, columnspan=8, sticky='n', padx=10, pady=10)

    def sensor_dropd_objID_clicked(self, choice):
        print("sensor objID dropdown clicked:", choice)
        #display the corresponding object type and attributes of the selected object id
        object_type, object_attributes = get_object_type_and_attributes(self.session_path, choice)
        self.sensor_f_details_f_objType_dropd.configure(variable=customtkinter.StringVar(value=object_type))
        self.sensor_f_details_f_attr_name_list = customtkinter.CTkTextbox(self.sensor_f_details_f, height=125)
        self.sensor_f_details_f_attr_value_list = customtkinter.CTkTextbox(self.sensor_f_details_f, height=125)
        for attribute in object_attributes:
            self.sensor_f_details_f_attr_name_list.insert("end", attribute["name"] + "\n") #index is 'line.character'
            self.sensor_f_details_f_attr_value_list.insert("end", attribute["value"] + "\n") #index is 'line.character'
        self.sensor_f_details_f_attr_name_list.grid(row=7, column=3, padx=(0,5), pady=(0,5))
        self.sensor_f_details_f_attr_value_list.grid(row=7, column=4, padx=(0,20), pady=(0,5))

    def add_related_object(self):
        curr_rel_obj = self.sensor_f_details_f_relObj_dropd.get()
        if curr_rel_obj != "":
            self.sensor_f_details_f_relObj_list.insert("end", curr_rel_obj + "\n")

    def str_represents_int(self, string):
        try: 
            int(string)
        except ValueError:
            return False
        else:
            return True
        
    def str_represents_float(self, string):
        try: 
            float(string)
        except ValueError:
            return False
        else:
            return True

    def get_obj_attributes_list_sensor(self):
        attr_list = []
        textBox_names = self.sensor_f_details_f_attr_name_list.get(0.0, "end")
        allNames = textBox_names.split(sep="\n")
        textBox_values = self.sensor_f_details_f_attr_value_list.get(0.0, "end")
        allValues = textBox_values.split(sep="\n")
        for index, attr_name in enumerate(allNames):
            if not attr_name == "":
                curr_attribute = {
                    "name":attr_name,
                    "value":allValues[index],
                    "time":""
                }
                attr_list.append(curr_attribute)
        return attr_list

    def continuous_extract_events(self):
        #print("continuous_extract_events")
        timeColumn_value_raw = self.sensor_f_details_f_time_textbox.get(1.0, "end")
        timeColumn_value = timeColumn_value_raw.split(sep="\n")[0]
        valueColumn_value_raw = self.sensor_f_details_f_value_textbox.get(1.0, "end")
        valueColumn_value = valueColumn_value_raw.split(sep="\n")[0]
        if timeColumn_value == "":
            self.sensor_f_continuous_f_extract_events_button_feedback.configure(text="No timestamp column given.", text_color="red")
            return
        elif valueColumn_value == "":
            self.sensor_f_continuous_f_extract_events_button_feedback.configure(text="No sensor values column given.", text_color="red")
            return
        if not timeColumn_value in self.df_sensor_file.columns.values:
            self.sensor_f_continuous_f_extract_events_button_feedback.configure(text="Time column not found in .csv file.", text_color="red")
            return
        if not valueColumn_value in self.df_sensor_file.columns.values:
            self.sensor_f_continuous_f_extract_events_button_feedback.configure(text="Sensor values column not found in .csv file.", text_color="red")
            return
        if self.sensor_f_details_f_objID_dropd.get() == "":
            self.sensor_f_continuous_f_extract_events_button_feedback.configure(text="No Object ID selected.", text_color="red")
            return
        if self.sensor_f_details_f_objType_dropd.get() == "":
            self.sensor_f_continuous_f_extract_events_button_feedback.configure(text="No Object Type selected.", text_color="red")
            return
        relObjects_values = self.sensor_f_details_f_relObj_list.get(0.0, "end")
        allRelObjects = relObjects_values.split(sep="\n")
        allRelObjects_wo_empty = list(filter(None, allRelObjects))
        for relObj in allRelObjects_wo_empty:
            if relObj not in get_all_object_ids(self.session_path):
                self.sensor_f_continuous_f_extract_events_button_feedback.configure(text=f"Related object {relObj} is not a known object.", text_color="red")
                return
        if self.sensor_f_details_f_objID_dropd.get() in allRelObjects:
            self.sensor_f_continuous_f_extract_events_button_feedback.configure(text="Object ID is not allowed in Related Objects list.", text_color="red")
            return
        
        lastXEntries_greater_value = self.sensor_f_continuous_f_lastXEntries_greater_text.get(1.0, "end").split(sep="\n")[0]
        if not self.str_represents_int(lastXEntries_greater_value):
            self.sensor_f_continuous_f_extract_events_button_feedback.configure(text="Value of change in last X at > must be a number >= 1.", text_color="red")
            return
        if int(lastXEntries_greater_value) < 1:
            self.sensor_f_continuous_f_extract_events_button_feedback.configure(text="Value of change in last X at > must be >= 1.", text_color="red")
            return
        lastXEntries_smaller_value = self.sensor_f_continuous_f_lastXEntries_smaller_text.get(1.0, "end").split(sep="\n")[0]
        if not self.str_represents_int(lastXEntries_smaller_value):
            self.sensor_f_continuous_f_extract_events_button_feedback.configure(text="Value of change in last X at < must be an integer >= 1.", text_color="red")
            return
        if int(lastXEntries_smaller_value) < 1:
            self.sensor_f_continuous_f_extract_events_button_feedback.configure(text="Value of change in last X at < must be >= 1.", text_color="red")
            return
        threshold_greater_value = self.sensor_f_continuous_f_threshold_greater_text.get(1.0, "end").split(sep="\n")[0]
        if not self.str_represents_float(threshold_greater_value):
            self.sensor_f_continuous_f_extract_events_button_feedback.configure(text="Value of threshold at > must be a number > 0.0.", text_color="red")
            return
        if float(threshold_greater_value) <= 0.0:
            self.sensor_f_continuous_f_extract_events_button_feedback.configure(text="Value of threshold at > must be > 0.0.", text_color="red")
            return
        threshold_smaller_value = self.sensor_f_continuous_f_threshold_smaller_text.get(1.0, "end").split(sep="\n")[0]
        if not self.str_represents_float(threshold_smaller_value):
            self.sensor_f_continuous_f_extract_events_button_feedback.configure(text="Value of threshold at < must be a number < 0.0.", text_color="red")
            return
        if float(threshold_smaller_value) >= 0.0:
            self.sensor_f_continuous_f_extract_events_button_feedback.configure(text="Value of threshold at < must be < 0.0.", text_color="red")
            return
        activity_name_greater = self.sensor_f_continuous_f_activityName_greater_text.get(1.0, "end").split(sep="\n")[0]
        if activity_name_greater == "":
            self.sensor_f_continuous_f_extract_events_button_feedback.configure(text="No activity name given.", text_color="red")
            return
        activity_name_smaller = self.sensor_f_continuous_f_activityName_smaller_text.get(1.0, "end").split(sep="\n")[0]
        if activity_name_smaller == "":
            self.sensor_f_continuous_f_extract_events_button_feedback.configure(text="No activity name given.", text_color="red")
            return
        
        #extract events from file
        events = extract_events_continuous_data(self.df_sensor_file, self.curr_sensor_file_name, timeColumn_value, valueColumn_value,
                                                self.sensor_f_details_f_objID_dropd.get(), allRelObjects_wo_empty,int(lastXEntries_greater_value),
                                                float(threshold_greater_value), activity_name_greater, int(lastXEntries_smaller_value),
                                                float(threshold_smaller_value), activity_name_smaller)
        add_events(self.session_path, events)

        #if success in extracting events: If object ID was a new one, add it to final_ocel2.json
        curr_attr_list = self.get_obj_attributes_list_sensor()
        update_or_create_object(self.session_path, self.sensor_f_details_f_objID_dropd.get(), self.sensor_f_details_f_objType_dropd.get(), curr_attr_list)

        self.sensor_f_continuous_f_extract_events_button_feedback.configure(
            text="Successfully extracted events from continuous sensor file.\n Load new sensor file or finish step", text_color="green")

    def discrete_extract_events(self):
        print("discrete_extract_events")
        timeColumn_value_raw = self.sensor_f_details_f_time_textbox.get(1.0, "end")
        timeColumn_value = timeColumn_value_raw.split(sep="\n")[0]
        valueColumn_value_raw = self.sensor_f_details_f_value_textbox.get(1.0, "end")
        valueColumn_value = valueColumn_value_raw.split(sep="\n")[0]
        if timeColumn_value == "":
            self.sensor_f_discrete_f_extract_events_button_feedback.configure(text="No timestamp column given.", text_color="red")
            return
        elif valueColumn_value == "":
            self.sensor_f_discrete_f_extract_events_button_feedback.configure(text="No sensor values column given.", text_color="red")
            return
        if not timeColumn_value in self.df_sensor_file.columns.values:
            self.sensor_f_discrete_f_extract_events_button_feedback.configure(text="Time column not found in .csv file.", text_color="red")
            return
        if not valueColumn_value in self.df_sensor_file.columns.values:
            self.sensor_f_discrete_f_extract_events_button_feedback.configure(text="Sensor values column not found in .csv file.", text_color="red")
            return
        if self.sensor_f_details_f_objID_dropd.get() == "":
            self.sensor_f_discrete_f_extract_events_button_feedback.configure(text="No Object ID selected.", text_color="red")
            return
        if self.sensor_f_details_f_objType_dropd.get() == "":
            self.sensor_f_discrete_f_extract_events_button_feedback.configure(text="No Object Type selected.", text_color="red")
            return
        relObjects_values = self.sensor_f_details_f_relObj_list.get(0.0, "end")
        allRelObjects = relObjects_values.split(sep="\n")
        allRelObjects_wo_empty = list(filter(None, allRelObjects))
        for relObj in allRelObjects_wo_empty:
            if relObj not in get_all_object_ids(self.session_path):
                self.sensor_f_discrete_f_extract_events_button_feedback.configure(text=f"Related object {relObj} is not a known object.", text_color="red")
                return
        if self.sensor_f_details_f_objID_dropd.get() in allRelObjects:
            self.sensor_f_discrete_f_extract_events_button_feedback.configure(text="Object ID is not allowed in Related Objects list.", text_color="red")
            return
        
        states_values = self.sensor_f_discrete_f_states_list.get(0.0, "end")
        all_State_values = states_values.split(sep="\n")
        all_State_values_wo_empty = list(filter(None, all_State_values))
        if len(all_State_values_wo_empty) == 0:
            self.sensor_f_discrete_f_extract_events_button_feedback.configure(text="Sensor states list is empty.", text_color="red")
            return
        activity_names_values = self.sensor_f_discrete_f_activityName_list.get(0.0, "end")
        all_activity_names = activity_names_values.split(sep="\n")
        all_activity_names_wo_empty = list(filter(None, all_activity_names))
        if len(all_State_values_wo_empty) != len(all_activity_names_wo_empty):
            self.sensor_f_discrete_f_extract_events_button_feedback.configure(text="Sensor states list and activity names list are not same length.", text_color="red")
            return
        
        #extract events from file
        events = extract_events_discrete_data(self.df_sensor_file, self.curr_sensor_file_name, timeColumn_value, valueColumn_value,
                                                self.sensor_f_details_f_objID_dropd.get(), allRelObjects_wo_empty, all_State_values_wo_empty, all_activity_names_wo_empty)
        add_events(self.session_path, events)

        #if success in extracting events: If object ID was a new one, add it to final_ocel2.json
        curr_attr_list = self.get_obj_attributes_list_sensor()
        update_or_create_object(self.session_path, self.sensor_f_details_f_objID_dropd.get(), self.sensor_f_details_f_objType_dropd.get(), curr_attr_list)

        self.sensor_f_discrete_f_extract_events_button_feedback.configure(
            text="Successfully extracted events from discrete sensor file.\n Load new sensor file or finish step", text_color="green")

    def finish_sensor_step(self):
        print("finish_sensor_step")
        self.set_status("sensor_objects_created", True)
        self.set_status("sensor_events_created", True)
        self.disable_page("all")
        self.enable_page("analysis")
        self.show_frame("analysis")
    
    
    #Analysis Functions
    def analysis_create_overall_ocel(self):
        #add objectTypes and eventTypes to final ocel by reading it in and writing it again)
        success, ocel = objects_events_from_ocel(self.session_path + "final_ocel2.json")
        if success == "Success":
            pm4py.write.write_ocel2_json(ocel, self.session_path + 'final_ocel2.json')


        self.analysis_f_button_createOCEL_feedback.configure(text="Successfully created final ocel file.", text_color="green")
        self.analysis_f_button_analyze.grid(row=4, column=1, padx=20, pady=(20, 10)) #Show button for analysis
        self.set_status("final_json_created", True)

    def analysis_analyze_overall_ocel(self):
        #analyze final ocel file
        ocel = pm4py.read_ocel2_json(self.session_path + "final_ocel2.json")
        #print(ocel)
        
        self.analysis_f_statistics_f.grid(row=5, column=1, sticky='n', padx=10, pady=10)

        self.analysis_f_statistics_f_numbObj_value.configure(text=str(len(get_all_object_ids(self.session_path))))
        
        numb_events, numb_video_events, numb_sensor_events = get_events_summary(self.session_path)
        self.analysis_f_statistics_f_numbEvents_value.configure(text=str(numb_events))
        self.analysis_f_statistics_f_numbEventsVideos_value.configure(text=str(numb_video_events))
        self.analysis_f_statistics_f_numbEventsSensors_value.configure(text=str(numb_sensor_events))
        
        object_types = get_all_object_types(self.session_path)
        for object_type in object_types:
            self.analysis_f_statistics_f_objTypes_values.insert("end", object_type + "\n")
        event_types = get_all_event_types(self.session_path)
        for event_type in event_types:
            self.analysis_f_statistics_f_EventTypes_values.insert("end", event_type + "\n")

        #Visualizations
        analyzeOCEL(ocel, "dfg_frequency", view=True, save=True, savepath=self.session_path + "results/") #methods: dfg_frequency, dfg_performance, petri_net
        analyzeOCEL(ocel, "dfg_performance", view=False, save=True, savepath=self.session_path + "results/")
        analyzeOCEL(ocel, "petri_net", view=False, save=True, savepath=self.session_path + "results/")


if __name__ == "__main__":
    app = App()
    app.mainloop()
