import pm4py
from pm4py.objects.ocel.validation import jsonocel
import json
import pandas as pd

def get_all_object_ids(session_path):
    """
    get_all_object_ids takes the current session path and returns a list with all object IDs in the 'final_ocel2.json' file of the current session

    :session_path: The path to the current session, including a '/' at the end
    :return: Returns a list with all object IDs in the 'final_ocel2.json' file of the current session
    """

    all_obj_ids = []
    with open(session_path + 'final_ocel2.json', "r") as ocel:
        data = json.load(ocel)
    
    for object in data["objects"]:
        all_obj_ids.append(object["id"])

    return all_obj_ids

def update_or_create_object(session_path, object_id, object_type, object_attributes):
    """
    update_or_create_object takes the current session path and all infos on the current object (id, type, attributes)
        and looks if the object already exists in the final_ocel2.json. If it exists the data is updated, if not the object is newly created.
        If object_id is "" nothing is done.

    :session_path: The path to the current session, including a '/' at the end
    :object_id: The object ID
    :object_type: The object type
    :object_attributes: The object attributes as list
    :return:
    """
    if object_id == "":
        return

    with open(session_path + 'final_ocel2.json', "r") as ocel_in:
        data = json.load(ocel_in)
    
    obj_already_existed = False
    for object in data["objects"]:
        if object["id"] == object_id:
            object["type"] = object_type
            object["attributes"] = object_attributes
            print("Updated object type and attributes of existing object.")
            obj_already_existed = True
    
    if obj_already_existed == False:
        #create new object
        new_object = {
                    "id":object_id,
                    "type":object_type,
                    "attributes":object_attributes,
                    "relationships":[]
                    }
        data["objects"].append(new_object)
        print("Created new object.")
    
    with open(session_path + 'final_ocel2.json', "w") as ocel_out:
        json.dump(data, ocel_out)

def get_object_type_and_attributes(session_path, object_id):
    """
    get_object_type_and_attributes takes the current session path and object id and gets the corresponding object type and attributes (as list of dicts) from the final_ocel2.json

    :session_path: The path to the current session, including a '/' at the end
    :object_id: The object ID
    :return: object type as string and object attributes as list of dicts
    """

    with open(session_path + 'final_ocel2.json', "r") as ocel_in:
        data = json.load(ocel_in)
    for object in data["objects"]:
        if object["id"] == object_id:
            object_type = object["type"]
            object_attributes = object["attributes"]

    return object_type, object_attributes

def get_all_object_types(session_path):
    """
    get_all_object_types takes the current session path and returns a list with all object types (unique) in the 'final_ocel2.json' file of the current session

    :session_path: The path to the current session, including a '/' at the end
    :return: Returns a list with all object types (unique) in the 'final_ocel2.json' file of the current session
    """

    all_obj_types = []
    with open(session_path + 'final_ocel2.json', "r") as ocel:
        data = json.load(ocel)
    
    for object in data["objects"]:
        all_obj_types.append(object["type"])

    all_obj_types = list(set(all_obj_types))
    return all_obj_types

def get_all_event_ids(session_path):
    """
    get_all_event_ids takes the current session path and returns a list with all event IDs in the 'final_ocel2.json' file of the current session

    :session_path: The path to the current session, including a '/' at the end
    :return: Returns a list with all event IDs in the 'final_ocel2.json' file of the current session
    """

    all_event_ids = []
    with open(session_path + 'final_ocel2.json', "r") as ocel:
        data = json.load(ocel)
    
    for event in data["events"]:
        all_event_ids.append(event["id"])

    return all_event_ids

def get_all_event_types(session_path):
    """
    get_all_event_types takes the current session path and returns a list with all event types (unique) in the 'final_ocel2.json' file of the current session

    :session_path: The path to the current session, including a '/' at the end
    :return: Returns a list with all event types (unique) in the 'final_ocel2.json' file of the current session
    """

    all_event_types = []
    with open(session_path + 'final_ocel2.json', "r") as ocel:
        data = json.load(ocel)
    
    for event in data["events"]:
        all_event_types.append(event["type"])

    all_event_types = list(set(all_event_types))
    return all_event_types

def get_events_summary(session_path):
    """
    get_events_summary takes the current session path and returns statistics on events

    :session_path: The path to the current session, including a '/' at the end
    :return: Returns the number of events, number of video events, and number of sensor events as integers
    """

    numb_events, numb_video_events, numb_sensor_events = 0, 0, 0
    with open(session_path + 'final_ocel2.json', "r") as ocel:
        data = json.load(ocel)
    
    for event in data["events"]:
        numb_events += 1
        attributes = event["attributes"]
        for attribute in attributes:
            if attribute["name"] == "event_source":
                if attribute["value"] == "video":
                    numb_video_events += 1
                elif attribute["value"] == "sensor":
                    numb_sensor_events += 1

    return numb_events, numb_video_events, numb_sensor_events

def add_events(session_path, events):
    """
    add_events takes the current session path and all infos on new events and looks if the event already exists in the final_ocel2.json.
        If it does not exist the event is newly created.

    :session_path: The path to the current session, including a '/' at the end
    :events: A list of event data with each event in this format:
        {"id": "event 1","type": "event name","time": "1980-01-01T10:28:00.000000",
        "attributes": [{"name":"attr_name","value":"90"}],
        "relationships": [{"objectId": "object_1","qualifier": ""}]}
    :return:
    """
    with open(session_path + 'final_ocel2.json', "r") as ocel_in:
        data = json.load(ocel_in)
    for event in events:
        if event != "":
            curr_event_id = event["id"]
            event_already_existed = False
            for json_file_events in data["events"]:
                if json_file_events["id"] == curr_event_id:
                    event_already_existed = True
            
            if event_already_existed == False:
                #create new event
                data["events"].append(event)
                #print("Created new event.")
            
    with open(session_path + 'final_ocel2.json', "w") as ocel_out:
        json.dump(data, ocel_out)
    print("Created new events.")

if __name__ == "__main__":


