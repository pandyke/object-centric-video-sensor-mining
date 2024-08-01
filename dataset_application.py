import csv
import cv2
import pm4py
from pm4py.objects.ocel.validation import jsonocel
import json
import pandas as pd
import os

from analysis import get_all_object_types
from ocel_utilities import analyzeOCEL, get_ocel_from_ocel1_or_ocel2
from video_processing import init_object_tracking, manual_object_definer_prep, object_labeling_image_prep, object_tracking

def convert_gt_to_ocel2(data_path, scenes):
    for scene in scenes:
        gt_ocel_path = data_path + scene + "/" + scene + "_overall_ocel.jsonocel"
        ocel1 = get_ocel_from_ocel1_or_ocel2(gt_ocel_path)
        path_converted_to_ocel2 = data_path + scene + "/" + scene + "_overall_ocel2.json"
        pm4py.write.write_ocel2_json(ocel1, path_converted_to_ocel2)

def postprocess_gt(data_path, scenes):
    for scene in scenes:
        gt_ocel2_path = data_path + scene + "/" + scene + "_overall_ocel2.json"

        replace_dict = {}
        if scene == "scene02":
            replace_dict = {"Pick asset from warehouse":"Pick asset from warehouse/Move asset to storage",
                            "XX":"Move asset to storage/Move asset to storage",
                            "Unpack asset":"Unpack asset/Test asset quality and functionality/Install and configure asset/Create system entry for asset",
                            "Test asset quality and functionality":"Unpack asset/Test asset quality and functionality/Install and configure asset/Create system entry for asset",
                            "Install and configure asset":"Unpack asset/Test asset quality and functionality/Install and configure asset/Create system entry for asset",
                            "Create system entry for asset":"Unpack asset/Test asset quality and functionality/Install and configure asset/Create system entry for asset",
                            "Pick asset from self service desk":"Pick asset from self service desk/Move asset to storage",
                            "Move asset to storage":"Pick asset from self service desk/Move asset to storage"}
        elif scene == "scene03":
            replace_dict = {"Pick asset from warehouse":"Pick asset from warehouse/Move asset to storage",
                            "Move asset to storage":"Pick asset from warehouse/Move asset to storage",
                            "Unpack asset":"Unpack asset/Test asset quality and functionality/Install and configure asset/Move asset to repair desk/Label asset",
                            "Test asset quality and functionality":"Unpack asset/Test asset quality and functionality/Install and configure asset/Move asset to repair desk/Label asset",
                            "Install and configure asset":"Unpack asset/Test asset quality and functionality/Install and configure asset/Move asset to repair desk/Label asset",
                            "Move asset to repair desk":"Unpack asset/Test asset quality and functionality/Install and configure asset/Move asset to repair desk/Label asset",
                            "Label asset":"Unpack asset/Test asset quality and functionality/Install and configure asset/Move asset to repair desk/Label asset",
                            "Create system entry for asset":"Create system entry for asset/Check asset quality/Check-In asset for repair/Check-Out asset to user/Handover asset to user",
                            "Check asset quality":"Create system entry for asset/Check asset quality/Check-In asset for repair/Check-Out asset to user/Handover asset to user",
                            "Check-In asset for repair":"Create system entry for asset/Check asset quality/Check-In asset for repair/Check-Out asset to user/Handover asset to user",
                            "Check-Out asset to user":"Create system entry for asset/Check asset quality/Check-In asset for repair/Check-Out asset to user/Handover asset to user",
                            "Handover asset to user":"Create system entry for asset/Check asset quality/Check-In asset for repair/Check-Out asset to user/Handover asset to user",
                            "Enter room":"Enter room/Leave room",
                            "Leave room":"Enter room/Leave room"}
        elif scene == "scene04":
            replace_dict = {"Pick asset from self service desk":"Pick asset from self service desk/Pick asset from warehouse/Move asset to storage",
                            "Pick asset from warehouse":"Pick asset from self service desk/Pick asset from warehouse/Move asset to storage",
                            "Move asset to storage":"Pick asset from self service desk/Pick asset from warehouse/Move asset to storage",
                            "Unpack asset":"Unpack asset/Test asset quality and functionality/Install and configure asset/Label asset",
                            "Test asset quality and functionality":"Unpack asset/Test asset quality and functionality/Install and configure asset/Label asset",
                            "Install and configure asset":"Unpack asset/Test asset quality and functionality/Install and configure asset/Label asset",
                            "Label asset":"Unpack asset/Test asset quality and functionality/Install and configure asset/Label asset",
                            "Enter room":"Enter room/Leave room",
                            "Leave room":"Enter room/Leave room",
                            "Check asset quality":"Check asset quality/Check-In asset to storage",
                            "Check-In asset to storage":"Check asset quality/Check-In asset to storage"}
        elif scene == "scene05":
            replace_dict = {"Carry out repair":"Carry out repair/Check asset quality/Update asset status in the system/other, see notes",
                            "Check asset quality":"Carry out repair/Check asset quality/Update asset status in the system/other, see notes",
                            "Update asset status in the system":"Carry out repair/Check asset quality/Update asset status in the system/other, see notes",
                            "other, see notes":"Carry out repair/Check asset quality/Update asset status in the system/other, see notes",
                            "Enter room":"Enter room/Leave room",
                            "Leave room":"Enter room/Leave room"}
        elif scene == "scene06":
            replace_dict = {"Pick asset from warehouse":"Pick asset from warehouse/Move asset to storage",
                            "Move asset to storage":"Pick asset from warehouse/Move asset to storage",
                            "Unpack asset":"Unpack asset/Test asset quality and functionality/Install and configure asset/Label asset",
                            "Test asset quality and functionality":"Unpack asset/Test asset quality and functionality/Install and configure asset/Label asset",
                            "Install and configure asset":"Unpack asset/Test asset quality and functionality/Install and configure asset/Label asset",
                            "Label asset":"Unpack asset/Test asset quality and functionality/Install and configure asset/Label asset",
                            "Enter room":"Enter room/Leave room",
                            "Leave room":"Enter room/Leave room",
                            "Create system entry for asset":"Create system entry for asset/Check asset quality/Check-In asset to storage",
                            "Check asset quality":"Create system entry for asset/Check asset quality/Check-In asset to storage",
                            "Check-In asset to storage":"Create system entry for asset/Check asset quality/Check-In asset to storage"}

        with open(gt_ocel2_path, "r") as ocel_json:
            data = json.load(ocel_json)
        
        for event in data["events"]:
            if event["type"] in replace_dict:
                event["type"] = replace_dict[event["type"]]


        path_results = data_path + scene + "/" + scene + "_overall_ocel2_postprocessed.json"
        with open(path_results, "w") as ocel_out:
            json.dump(data, ocel_out)

def remove_repeating_activities_gt(data_path, scenes):
    for scene in scenes:
        gt_ocel2_path = data_path + scene + "/" + scene + "_overall_ocel2_postprocessed.json"
        with open(gt_ocel2_path, "r") as ocel_json:
            data = json.load(ocel_json)
        
        prev_activity = ""
        new_data_events = []
        for event in data["events"]:
            curr_activity = event["type"]
            if curr_activity != prev_activity:
                new_data_events.append(event)
            prev_activity = curr_activity
        data["events"] = new_data_events

        for event in data["events"]:
            for relationship in event["relationships"]:
                relationship["qualifier"] = ""

        path_results = data_path + scene + "/" + scene + "_overall_ocel2_rep_rem.json"
        with open(path_results, "w") as ocel_out:
            json.dump(data, ocel_out)

def remove_repeating_activities_res(results_path, scenes):
    for scene in scenes:
        res_ocel2_path = results_path + scene + "/" + scene + "_final_ocel2_finished.json"
        with open(res_ocel2_path, "r") as ocel_json:
            data = json.load(ocel_json)
        
        prev_activity = ""
        new_data_events = []
        for event in data["events"]:
            curr_activity = event["type"]
            if curr_activity != prev_activity:
                new_data_events.append(event)
            prev_activity = curr_activity
        data["events"] = new_data_events

        path_results = results_path + scene + "/" + scene + "_final_ocel2_rep_rem.json"
        with open(path_results, "w") as ocel_out:
            json.dump(data, ocel_out)

def get_all_object_types_gt(data_path):
    all_object_types = []
    scenes = ["scene02", "scene03", "scene04", "scene05", "scene06"]
    for scene in scenes:
        gt_ocel2_path = data_path + scene + "/" + scene + "_overall_ocel2_rep_rem.json"
        with open(gt_ocel2_path, "r") as ocel_json:
            data = json.load(ocel_json)
        for object in data["objects"]:
            all_object_types.append(object["type"])

    all_object_types = list(set(all_object_types))
    return all_object_types


def flatten_ocel(data_path, scenes, results_path, all_object_types):
    for scene in scenes:
        gt_ocel2_path = data_path + scene + "/" + scene + "_overall_ocel2_rep_rem.json"
        res_ocel2_path = results_path + scene + "/" + scene + "_final_ocel2_rep_rem.json"
        gt_flattened_logs_path = data_path + scene + "/flattened_logs/"
        res_flattened_logs_path = results_path + scene + "/flattened_logs/"

        for object_type in all_object_types:
            gt_ocel = get_ocel_from_ocel1_or_ocel2(gt_ocel2_path)
            gt_flattened_log = pm4py.ocel_flattening(gt_ocel, object_type)
            pm4py.write_xes(gt_flattened_log, gt_flattened_logs_path + scene + "_flattened_" + object_type + ".xes")

            res_ocel = get_ocel_from_ocel1_or_ocel2(res_ocel2_path)
            res_flattened_log = pm4py.ocel_flattening(res_ocel, object_type)
            pm4py.write_xes(res_flattened_log, res_flattened_logs_path + scene + "_flattened_" + object_type + ".xes")


def get_fitness(data_path, scenes, results_path, all_object_types):
    
    results_list = []
    for scene in scenes:
        path_gt_flattened_logs = data_path + scene + "/flattened_logs/"
        path_res_flattened_logs = results_path + scene + "/flattened_logs/"
        
        for object_type in all_object_types:
            curr_filename = scene + "_flattened_" + object_type + ".xes"
            gt_path_curr_flat_log = path_gt_flattened_logs + curr_filename
            log_gt = pm4py.read_xes(gt_path_curr_flat_log)
            if len(log_gt) > 0:
                log_gt = pm4py.convert_to_event_log(log_gt)
                net_gt, initial_marking_gt, final_marking_gt = pm4py.discover_petri_net_inductive(log_gt)

            res_path_curr_flat_log = path_res_flattened_logs + curr_filename
            log_res = pm4py.read_xes(res_path_curr_flat_log)
            if len(log_res) > 0:
                log_res = pm4py.convert_to_event_log(log_res)

            if len(log_gt) > 0 and len(log_res) > 0:
                fitness_token_based = pm4py.fitness_token_based_replay(log_res, net_gt, initial_marking_gt, final_marking_gt)
                results_list.append([object_type, scene, fitness_token_based["log_fitness"]])
                #print("fitness_token_based", fitness_token_based)

    results_df = pd.DataFrame(results_list, columns=['object_type', "scene",
                                                         'log_fitness_token_based'])

    return results_df


if __name__ == "__main__":
    data_path = "data/Solve4X_Application/"
    results_path = "results/Solve4X_Application/"
    scenes = ["scene02", "scene03", "scene04", "scene05", "scene06"]
    
    convert_gt_to_ocel2(data_path, scenes)
    postprocess_gt(data_path, scenes)
    remove_repeating_activities_gt(data_path, scenes)
    remove_repeating_activities_res(results_path, scenes)
    all_object_types = get_all_object_types_gt(data_path)
    flatten_ocel(data_path, scenes, results_path, all_object_types)
    results_df = get_fitness(data_path, scenes, results_path, all_object_types)
    print(results_df)
    results_df.to_excel(results_path + "fitness_all_scenes.xlsx")



