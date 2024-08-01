import pm4py
from pm4py.objects.ocel.validation import jsonocel
import json
import pandas as pd
from datetime import datetime
import os
import uuid


def preprocessProcessLog(file_path, session_path):
    """
    Customizable function.
    preprocessProcessLog prepares the process log for the next steps that extract relevant information for object-centric process mining.
        Columns are renamed to specific formats (=with specific prefixes) for further processing.
        The timestamps are converted to the right formats.
        In this function, e.g., a column could be split if it includes more than one relevant attribute, etc.
        The preprocessed process log is saved as a dataframe (.pkl) in the current session folder as originalFileName_preprocessed.pkl

    :file_path: The path to the file containing the process log. Format of the file can also be specified in the code, e.g., CSV
    :session_path: The path to the current session folder including a '/' at the end.
        Note: In the session folder the subfolder 'process_logs_preprocessed' has to exist for this function to work
    :return: Returns the string "Success" or a string containing the error
    """
    if not file_path.endswith('.csv'):
        return "File is not a .csv file. Function aborted"
    
    try:
        df_processLog = pd.read_csv(file_path, sep=",", dtype={'Serial': str})

        #column prefixes: comTimestamp:, comObjectType:, comEvent, comObjAttr:NAMEofOBJECTTYPE:, comEventAttr:
        df_processLog.rename(columns={'Time': 'comTimestamp:timestamp', 'Admin': 'comObjectType:asset_issuing_person', 'Action': 'comEvent', 'Status': 'comEventAttr:Status',
                                    'Item': 'comObjectType:it_asset', 'Serial': 'comObjAttr:it_asset:Serial', 'Target': 'comObjectType:asset_receiving_person',
                                    'eventID': 'comEventID'}, inplace=True)

        #Tranform timestamps to desired format
        df_processLog['comTimestamp:timestamp'] = pd.to_datetime(df_processLog['comTimestamp:timestamp'], format="ISO8601", utc=True)
        df_processLog['comTimestamp:timestamp'] = df_processLog["comTimestamp:timestamp"].dt.strftime('%Y-%m-%dT%H:%M:%S.%f%z')

        path_head, path_tail = os.path.split(file_path)
        file_name = path_tail.split(".")[0]

        saving_path = session_path + "process_logs_preprocessed/" + file_name + "_preprocessed.pkl"
        df_processLog.to_pickle(saving_path)

        return "Success"
    
    except Exception as err:
        return "Error in function 'preprocessProcessLog': " + str(repr(err))

def processLogToOCEL(file_path_prepr):
    """
    processLogToOCEL extracts objects and events from a preprocessed process log that contains specific column prefixes.
        Column prefixes: comTimestamp:, comObjectType:, comEvent, comObjAttr:NAMEofOBJECTTYPE:, comEventAttr:, comEventID
        Columns without a prefix are ignored/dismissed
        Note that a comObjAttr is defined as an attribute (and value) for a specific object, i.e.,
        if there are more data entries with that specific object, the attribute must, by definition, have the same value in every entry
        The function also adds a custom event attribute to each event for later identification of the source: {"event_source": "process_log"}

    :file_path_prepr: The path to the preprocessed process log
    :return: Returns a success string either containing "Success" or the error that occured as string,
        as well as the generated ocel data (json data) containing all events and objects
    """    

    if not file_path_prepr.endswith('.pkl'):
        print("File is not a .pkl file. Function aborted")
        return "File is not a .pkl file. Function aborted", None
    df_prepr_processLog = pd.read_pickle(file_path_prepr)

    #create new ocel2 data as basis
    with open('data/ocel2_template.json', "r") as ocel_template:
        ocel_json_data = json.load(ocel_template)

    #extract objects
    all_comObjectType_cols = [col for col in df_prepr_processLog if col.startswith('comObjectType:')] #get all object columns. E.g., comObjectType:it_asset
    all_objects = []
    for col in all_comObjectType_cols:
        object_type = col.split("comObjectType:",1)[1] #get object type, e.g., it_asset
        #get corresponding comObjAttr columns for current ObjectType
        all_comObjAttr_cols = [col for col in df_prepr_processLog if col.startswith('comObjAttr:'+object_type)] #e.g., comObjAttr:it_asset:Serial
        #only keep ObjectType column (e.g., comObjectType:it_asset) and corresponding comObjAttr columns (e.g., comObjAttr:it_asset:Serial)
        keeper_cols = [col] + all_comObjAttr_cols
        df_curr_objType = df_prepr_processLog[keeper_cols].copy()
        df_curr_objType.drop_duplicates(subset=[col], inplace=True) #make sure to add each object just once to the df_objects
        df_curr_objType.dropna(subset=[col], inplace=True, axis=0) #drop objects with na value
        for index, row in df_curr_objType.iterrows(): #iterate over unique objects and get corresponding attribute values
            object_id = row[col]
            curr_attr_list = []
            for comObjAttr in all_comObjAttr_cols: #iterate over all attributes of that object
                attr_name = comObjAttr.split("comObjAttr:"+object_type+":",1)[1] #get attribute name, e.g., Serial
                attr_value = row[comObjAttr] #get attribute value
                curr_attr_dict = {"name":attr_name, "value":attr_value, "time":""} #add attribute key+value to dictionary
                curr_attr_list.append(curr_attr_dict)
            #as a standard attribute save for later that this object initially was found in the process log
            curr_attr_list.append({"name":"object_origin", "value":"process_log", "time":""})
            
            curr_obj_dict = {"id":object_id, "type":object_type, "attributes":curr_attr_list, "relationships":[]}
            all_objects.append(curr_obj_dict)        
    ocel_json_data["objects"] = all_objects

    #extract events
    timestamp_col_name = [col for col in df_prepr_processLog if col.startswith('comTimestamp:')] #get columns that have the timestamp prefix
    if len(timestamp_col_name) != 1:
        print("Error: There must be exactly one column with the timestamp prefix. Function is aborted")
        return "Error: There must be exactly one column with the timestamp prefix. Function is aborted", None
    
    eventID_col_name = [col for col in df_prepr_processLog if col.startswith('comEventID')] #get columns that have the event ID prefix
    if len(eventID_col_name) != 1:
        print("Error: There must be exactly one column with the event ID prefix. Function is aborted")
        return "Error: There must be exactly one column with the event ID prefix. Function is aborted", None
    
    all_events= []
    all_comEventAttr_cols = [col for col in df_prepr_processLog if col.startswith('comEventAttr:')] #get all comEventAttr: columns. E.g., comEventAttr:Status
    for index, row in df_prepr_processLog.iterrows(): #iterate over events
        curr_timestamp = row[timestamp_col_name[0]]
        curr_eventID = row[eventID_col_name[0]]
        #add event attributes
        curr_attr_list = []
        for comEventAttr in all_comEventAttr_cols: #iterate over all attribute columns of that event
            attr_name = comEventAttr.split("comEventAttr:",1)[1] #get attribute name, e.g., Status
            attr_value = row[comEventAttr] #get attribute value
            curr_attr_dict = {"name":attr_name, "value":attr_value} #add attribute key+value to dictionary
            curr_attr_list.append(curr_attr_dict)
        curr_attr_list.append({"name":"event_origin", "value":"process_log"}) #manually also add {"event_origin": "process_log"} as event attribute
        #add the related objects to the events by looking in comObjectType: columns
        curr_rel_objects = []
        used_obj_ids = []
        for obj_column in all_comObjectType_cols: #iterate over all columns that inherit objects
            curr_obj_id = row[obj_column]
            if curr_obj_id != '' and not pd.isnull(curr_obj_id):
                if curr_obj_id not in used_obj_ids: #make sure that every related object is just added once
                    curr_rel_obj_dict = {"objectId": curr_obj_id, "qualifier": "" }
                    curr_rel_objects.append(curr_rel_obj_dict)
                    used_obj_ids.append(curr_obj_id)
        #take the event that happended as ocel:activity
        activity = row['comEvent']

        curr_event_dict = {"id":curr_eventID, "type":activity, "time":curr_timestamp, "attributes":curr_attr_list, "relationships":curr_rel_objects}
        all_events.append(curr_event_dict)

    ocel_json_data["events"] = all_events

    return "Success", ocel_json_data

def objects_events_from_ocel(file_path):
    """
    objects_events_from_ocel checks validity for ocel 2.0 format and returns the read in ocel.

    :file_path: The path to the ocel file
    :return: Returns a success string either containing "Success" or the error that occured as string,
        as well as the ocel that was read in from the file
    """

    if jsonocel.apply(file_path, "data/ocel2_schema.json"):
        print("Is valid OCEL2 format")
        ocel = pm4py.read_ocel2_json(file_path)
        return "Success", ocel
    elif jsonocel.apply(file_path, "data/ocel1_schema.json"):
        print("Is valid OCEL1 format")
        if file_path.endswith("jsonocel"):
            ocel = pm4py.read_ocel(file_path)
            return "Success", ocel
        else:
            return_message = "Valid OCEL1 file has to end with .jsonocel"
            print(return_message)
            return return_message, None
    else:
        return_message = "File is not a valid OCEL1 or OCEL2 file. Please check format against the 'ocel1_schema.json' or 'ocel2_schema.json' file. Function aborted"
        print(return_message)
        return return_message,None



if __name__ == "__main__":

