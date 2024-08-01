import uuid
import pandas as pd
import numpy as np
import time
import datetime
import os.path
from pathlib import Path


def extract_events_continuous_data(sensor_df, sensor_file_name, time_column, sensor_values_column, object_id, related_objects_list,
                                   greater_value_window, greater_threshold, greater_activity_name,
                                   smaller_value_window, smaller_threshold, smaller_activity_name):
    """
    extract_events_continuous_data takes the sensor data as dataframe and relevant infos on the data, related objects, and event trigger rules,
        and returns a list of created events ready to be incorporated into an ocel file.

    :sensor_df: The dataframe containing the continuous sensor data
    :sensor_file_name: Name of the current sensor file
    :time_column: Name of the column containing the timestamps
    :sensor_values_column: Name of the column containing the sensor values
    :related_objects_list: List of related objects (Object IDs)
    :greater_value_window: (int) Window how many data points should be observed in the past for greater comparison to current value
    :greater_threshold: (positive float) The upper threshold for current value - value from greater_value_window data points ago
    :greater_activity_name: The name that should be given to the activity that results from breaching upper threshold
    :smaller_value_window: (int) Window how many data points should be observed in the past for smaller comparison to current value
    :smaller_threshold: (negative float) The lower threshold for current value - value from greater_value_window data points ago
    :smaller_activity_name: The name that should be given to the activity that results from breaching lower threshold
    :return: Returns a list of new events with every event in this format:
        {"id": "event 1","type": "event name","time": "1980-01-01T10:28:00.000000",
        "attributes": [{"name":"attr_name","value":"90"}],
        "relationships": [{"objectId": "object_1","qualifier": ""}]}
    """
    events_list = []

    #sensor_df[time_column] = pd.to_datetime(sensor_df[time_column], format="ISO8601", utc=True)
    #sensor_df[time_column] = sensor_df[time_column].dt.strftime('%Y-%m-%dT%H:%M:%S.%f%z')
    sensor_df = sensor_df.sort_values([time_column], ascending=True, ignore_index=True) #sort by timestamp
    source_attr = {"name":"event_source","value":"sensor"}
    source_attr_detail = {"name":"event_source_file","value":sensor_file_name}

    prev_value = sensor_df[sensor_values_column].iloc[0] #get first sensor value
    for index, row in sensor_df.iterrows(): #iterate over df rows
        curr_timestamp = row[time_column]
        curr_value = row[sensor_values_column]
        if index >= greater_value_window:
            prev_value = sensor_df.loc[sensor_df.index[index-greater_value_window], sensor_values_column]
            if curr_value > prev_value + greater_threshold:
                #event upper boundary breached
                curr_event = {}
                curr_event_attr_list = []
                curr_event_attr_list.append(source_attr) #add once
                curr_event_attr_list.append(source_attr_detail) #add once
                curr_event_attr_list.append({"name": "sensor_value_window", "value": str(greater_value_window)})
                curr_event_attr_list.append({"name": "sensor_threshold", "value": str(greater_threshold)})
                curr_event_attr_list.append({"name": "sensor_old_value", "value": str(prev_value)})
                curr_event_attr_list.append({"name": "sensor_new_value", "value": str(curr_value)})
                curr_event_rel_list = [{"objectId": object_id, "qualifier": ""}]
                for relObj in related_objects_list:
                    curr_event_rel_list.append({"objectId": relObj, "qualifier": ""})
                curr_event["id"] = greater_activity_name + "_" + str(uuid.uuid4())
                curr_event["type"] = greater_activity_name
                curr_event["time"] = curr_timestamp
                curr_event["attributes"] = curr_event_attr_list
                curr_event["relationships"] = curr_event_rel_list
                events_list.append(curr_event)
        if index >= smaller_value_window:
            prev_value = sensor_df.loc[sensor_df.index[index-greater_value_window], sensor_values_column]
            if curr_value < prev_value + smaller_threshold: #note: smaller_threshold is < 0
                #event lower boundary breached
                curr_event = {}
                curr_event_attr_list = []
                curr_event_attr_list.append(source_attr) #add once
                curr_event_attr_list.append(source_attr_detail) #add once
                curr_event_attr_list.append({"name": "sensor_value_window", "value": str(smaller_value_window)})
                curr_event_attr_list.append({"name": "sensor_threshold", "value": str(smaller_threshold)})
                curr_event_attr_list.append({"name": "sensor_old_value", "value": str(prev_value)})
                curr_event_attr_list.append({"name": "sensor_new_value", "value": str(curr_value)})
                curr_event_rel_list = [{"objectId": object_id, "qualifier": ""}]
                for relObj in related_objects_list:
                    curr_event_rel_list.append({"objectId": relObj, "qualifier": ""})
                curr_event["id"] = smaller_activity_name + "_" + str(uuid.uuid4())
                curr_event["type"] = smaller_activity_name
                curr_event["time"] = curr_timestamp
                curr_event["attributes"] = curr_event_attr_list
                curr_event["relationships"] = curr_event_rel_list
                events_list.append(curr_event)

    return events_list

def extract_events_discrete_data(sensor_df, sensor_file_name, time_column, sensor_values_column, object_id, related_objects_list,
                                   states_list, activity_names_list):
    """
    extract_events_discrete_data takes the sensor data as dataframe and relevant infos on the data, related objects, and event info,
        and returns a list of created events ready to be incorporated into an ocel file.

    :sensor_df: The dataframe containing the continuous sensor data
    :sensor_file_name: Name of the current sensor file
    :time_column: Name of the column containing the timestamps
    :sensor_values_column: Name of the column containing the sensor values
    :related_objects_list: List of related objects (Object IDs)
    :states_list: List of all possible sensor states
    :activity_names_list: List of corresponding activity names when these states occur
    :return: Returns a list of new events with every event in this format:
        {"id": "event 1","type": "event name","time": "1980-01-01T10:28:00.000000",
        "attributes": [{"name":"attr_name","value":"90"}],
        "relationships": [{"objectId": "object_1","qualifier": ""}]}
    """
    events_list = []

    #sensor_df[time_column] = pd.to_datetime(sensor_df[time_column], format="ISO8601", utc=True)
    #sensor_df[time_column] = sensor_df[time_column].dt.strftime('%Y-%m-%dT%H:%M:%S.%f%z')
    sensor_df = sensor_df.sort_values([time_column], ascending=True, ignore_index=True) #sort by timestamp
    source_attr = {"name":"event_source","value":"sensor"}
    source_attr_detail = {"name":"event_source_file","value":sensor_file_name}

    event_name_translator = dict(zip(states_list, activity_names_list))
    for index, row in sensor_df.iterrows(): #iterate over df rows
        curr_timestamp = row[time_column]
        curr_state = row[sensor_values_column]
        if curr_state not in event_name_translator:
            print(f"Sensor state {curr_state} not found in states list.")
        else:
            curr_activity_name = event_name_translator[curr_state]
            curr_event = {}
            curr_event_attr_list = []
            curr_event_attr_list.append(source_attr) #add once
            curr_event_attr_list.append(source_attr_detail) #add once
            curr_event_attr_list.append({"name": "sensor_value", "value": curr_state})
            curr_event_rel_list = [{"objectId": object_id, "qualifier": ""}]
            for relObj in related_objects_list:
                curr_event_rel_list.append({"objectId": relObj, "qualifier": ""})
            curr_event["id"] = curr_activity_name + "_" + str(uuid.uuid4())
            curr_event["type"] = curr_activity_name
            curr_event["time"] = curr_timestamp
            curr_event["attributes"] = curr_event_attr_list
            curr_event["relationships"] = curr_event_rel_list
            events_list.append(curr_event)

    return events_list


def eventTriggerDiscreteStandard(sensor_file_path, object_id, sensor_file_name):
    """
    eventTriggerDiscreteStandard is the standard algorithm that extracts events from a given sensor data file of a discrete sensor.
        It defines an event for every change of the sensor value, i.e., for every row in the .csv file.
        It assumes that the first column contains the timestamp in the format "%Y-%m-%d %H:%M:%S.%f" and the second column the discrete sensor values treated as string

    :sensor_file_path: The path to the sensor file as .csv, e.g., "data/sensor_discrete.csv".
    :object_id: The object ID that is assigned to the sensor. A sensor can be part of an object (e.g., of a machine) or can be an object itself
    :return: Returns a dataframe containing the extracted events.
        Format: ocel:eid, ocel:timestamp, ocel:activity, ocel:omap, ocel:vmap
    """

    df_sensor_data = pd.read_csv(sensor_file_path, sep=",")
    col_names = list(df_sensor_data.columns)
    if len(col_names) != 2:
        print("Discrete sensor csv file must contain exactly 2 columns: First column containing the timestamps, second column containg the sensor values. Function is aborted.")
        return
    df_sensor_data.rename(columns={col_names[0]: 'timestamp', col_names[1]: 'sensor_value'}, inplace=True)
    #format timestamp
    df_sensor_data["timestamp"] = pd.to_datetime(df_sensor_data["timestamp"], format="ISO8601", utc=True)
    df_sensor_data["timestamp"] = df_sensor_data["timestamp"].dt.strftime('%Y-%m-%dT%H:%M:%S.%f%z')

    events_list = []
    activities = df_sensor_data['sensor_value'].unique()
    rolling_number_activities = dict.fromkeys(activities, 1)
    for index, row in df_sensor_data.iterrows():
        curr_timestamp = row['timestamp']
        curr_rel_obj = [object_id]
        curr_event_attr = {"event_source": "sensor file " + sensor_file_name}
        curr_activity = row['sensor_value']
        rolling_event_id = rolling_number_activities.get(curr_activity)
        curr_event_id = curr_activity + "_" + str(rolling_event_id)
        rolling_number_activities[curr_activity] = rolling_event_id + 1
        events_list.append([curr_event_id, curr_timestamp, curr_activity, curr_rel_obj, curr_event_attr])

    df_events = pd.DataFrame(events_list, columns = ['ocel:eid', 'ocel:timestamp', 'ocel:activity', 'ocel:omap', 'ocel:vmap']) #create final events dataframe
    return df_events

def eventTriggerContinuousStandard(sensor_file_path, object_id, sensor_file_name):
    """
    eventTriggerContinuousStandard is the standard algorithm that extracts events from a given sensor data file of a continuous sensor.
        It calculates the mean value from all sensor values in the file and defines an event when the value rises above or below 1.5 standard deviations from the mean
        It assumes that the first column contains the timestamp in the format "%Y-%m-%d %H:%M:%S.%f" and the second column the continuous sensor values treated as double

    :sensor_file_path: The path to the sensor file as .csv, e.g., "data/sensor_continuous.csv".
    :object_id: The object ID that is assigned to the sensor. A sensor can be part of an object (e.g., of a machine) or can be an object itself
    :return: Returns a dataframe containing the extracted events.
        Format: ocel:eid, ocel:timestamp, ocel:activity, ocel:omap, ocel:vmap
    """

    df_sensor_data = pd.read_csv(sensor_file_path, sep=",")
    col_names = list(df_sensor_data.columns)
    if len(col_names) != 2:
        print("Continuous sensor csv file must contain exactly 2 columns: First column containing the timestamps, second column containg the sensor values. Function is aborted.")
        return
    df_sensor_data.rename(columns={col_names[0]: 'timestamp', col_names[1]: 'sensor_value'}, inplace=True)
    #format timestamp
    df_sensor_data["timestamp"] = pd.to_datetime(df_sensor_data["timestamp"], format="ISO8601", utc=True)
    df_sensor_data["timestamp"] = df_sensor_data["timestamp"].dt.strftime('%Y-%m-%dT%H:%M:%S.%f%z')

    events_list = []
    sensor_values = df_sensor_data['sensor_value'].tolist()
    values_mean = np.nanmean(sensor_values) #ignores nan values
    values_std = np.nanstd(sensor_values) #ignores nan values
    upper_boundary = values_mean + (1.5*values_std)
    lower_boundary = values_mean - (1.5*values_std)

    rolling_number_activities = dict.fromkeys(('sensor value above upper boundary', 'sensor value under lower boundary'), 1)
    prev_value = sensor_values[0]
    for index, row in df_sensor_data.iterrows():
        curr_value = row['sensor_value']
        if curr_value > upper_boundary and prev_value <= upper_boundary:
            prev_value = curr_value
            curr_activity = "sensor value above upper boundary"
            rolling_event_id = rolling_number_activities.get(curr_activity)
            curr_event_id = curr_activity + "_" + str(rolling_event_id)
            rolling_number_activities[curr_activity] = rolling_event_id + 1
        elif curr_value < lower_boundary and prev_value >= lower_boundary:
            prev_value = curr_value
            curr_activity = "sensor value under lower boundary"
            rolling_event_id = rolling_number_activities.get(curr_activity)
            curr_event_id = curr_activity + "_" + str(rolling_event_id)
            rolling_number_activities[curr_activity] = rolling_event_id + 1
        else:
            prev_value = curr_value
            continue
        curr_timestamp = row['timestamp']
        curr_rel_obj = [object_id]
        curr_event_attr = {"event_source": "sensor file " + sensor_file_name}
        events_list.append([curr_event_id, curr_timestamp, curr_activity, curr_rel_obj, curr_event_attr])

    df_events = pd.DataFrame(events_list, columns = ['ocel:eid', 'ocel:timestamp', 'ocel:activity', 'ocel:omap', 'ocel:vmap']) #create final events dataframe
    return df_events

def eventTriggerCustomAirQualitySensor(sensor_file_path, object_id, sensor_file_name):
    """
    eventTriggerCustomAirQualitySensor is a custom algorithm that extracts events from the air quality sensor data file which contains two values (temperature and humidity).
        It calculates the mean value from all sensor values in the file and defines an event when the value rises above or below 1.5 standard deviations from the mean
        It assumes that the first column contains the timestamp in the format "%Y-%m-%d %H:%M:%S.%f" and the second and third column the sensor values for temp and humidity as doubles

    :sensor_file_path: The path to the sensor file as .csv, e.g., "data/sensor_air_quality.csv".
    :object_id: The object ID that is assigned to the sensor. A sensor can be part of an object (e.g., of a machine) or can be an object itself
    :return: Returns a dataframe containing the extracted events.
        Format: ocel:eid, ocel:timestamp, ocel:activity, ocel:omap, ocel:vmap
    """

    df_sensor_data = pd.read_csv(sensor_file_path, sep=",")
    col_names = list(df_sensor_data.columns)
    df_sensor_data.rename(columns={col_names[0]: 'timestamp', col_names[1]: 'temperature', col_names[2]: 'humidity'}, inplace=True)
    #format timestamp
    df_sensor_data["timestamp"] = pd.to_datetime(df_sensor_data["timestamp"], format="ISO8601", utc=True)
    df_sensor_data["timestamp"] = df_sensor_data["timestamp"].dt.strftime('%Y-%m-%dT%H:%M:%S.%f%z')

    events_list = []
    sensor_values_temp = df_sensor_data['temperature'].tolist()
    sensor_values_hum = df_sensor_data['humidity'].tolist()
    values_mean_temp = np.nanmean(sensor_values_temp) #ignores nan values
    values_mean_hum = np.nanmean(sensor_values_hum) #ignores nan values
    values_std_temp = np.nanstd(sensor_values_temp) #ignores nan values
    values_std_hum = np.nanstd(sensor_values_hum) #ignores nan values
    upper_boundary_temp = values_mean_temp + (0.5*values_std_temp)
    lower_boundary_temp = values_mean_temp - (0.5*values_std_temp)
    upper_boundary_hum = values_mean_hum + (1.0*values_std_hum)
    lower_boundary_hum = values_mean_hum - (1.0*values_std_hum)

    rolling_number_activities = dict.fromkeys(('air quality above upper boundary', 'air quality under lower boundary'), 1)
    prev_value_temp = sensor_values_temp[0]
    prev_value_hum = sensor_values_hum[0]
    for index, row in df_sensor_data.iterrows():
        curr_value_temp = row['temperature']
        curr_value_hum = row['humidity']
        if curr_value_temp > upper_boundary_temp and prev_value_temp <= upper_boundary_temp and curr_value_hum > upper_boundary_hum and prev_value_hum <= upper_boundary_hum:
            prev_value_temp = curr_value_temp
            prev_value_hum = curr_value_hum
            curr_activity = "air quality above upper boundary"
            rolling_event_id = rolling_number_activities.get(curr_activity)
            curr_event_id = curr_activity + "_" + str(rolling_event_id)
            rolling_number_activities[curr_activity] = rolling_event_id + 1
        elif curr_value_temp < lower_boundary_temp and prev_value_temp >= lower_boundary_temp and curr_value_hum < lower_boundary_hum and prev_value_hum >= lower_boundary_hum:
            prev_value_temp = curr_value_temp
            prev_value_hum = curr_value_hum
            curr_activity = "air quality under lower boundary"
            rolling_event_id = rolling_number_activities.get(curr_activity)
            curr_event_id = curr_activity + "_" + str(rolling_event_id)
            rolling_number_activities[curr_activity] = rolling_event_id + 1
        else:
            prev_value_temp = curr_value_temp
            prev_value_hum = curr_value_hum
            continue
        curr_timestamp = row['timestamp']
        curr_rel_obj = [object_id]
        curr_event_attr = {"event_source": "sensor file " + sensor_file_name}
        events_list.append([curr_event_id, curr_timestamp, curr_activity, curr_rel_obj, curr_event_attr])

    df_events = pd.DataFrame(events_list, columns = ['ocel:eid', 'ocel:timestamp', 'ocel:activity', 'ocel:omap', 'ocel:vmap']) #create final events dataframe
    return df_events

def eventTriggerCustomDistanceSensor(sensor_file_path, object_id, sensor_file_name):
    """
    eventTriggerCustomDistanceSensor is a custom algorithm that extracts events from a given sensor data file of a distance sensor.
        It assumes that the first column contains the timestamp in the format "%Y-%m-%d %H:%M:%S.%f" and the second column the continuous sensor values treated as double

    :sensor_file_path: The path to the sensor file as .csv, e.g., "data/sensor_continuous.csv".
    :object_id: The object ID that is assigned to the sensor. A sensor can be part of an object (e.g., of a machine) or can be an object itself
    :return: Returns a dataframe containing the extracted events.
        Format: ocel:eid, ocel:timestamp, ocel:activity, ocel:omap, ocel:vmap
    """

    df_sensor_data = pd.read_csv(sensor_file_path, sep=",")
    col_names = list(df_sensor_data.columns)
    if len(col_names) != 2:
        print("Distance sensor csv file must contain exactly 2 columns: First column containing the timestamps, second column containg the sensor values. Function is aborted.")
        return
    df_sensor_data.rename(columns={col_names[0]: 'timestamp', col_names[1]: 'sensor_value'}, inplace=True)
    #format timestamp
    df_sensor_data["timestamp"] = pd.to_datetime(df_sensor_data["timestamp"], format="ISO8601", utc=True)
    df_sensor_data["timestamp"] = df_sensor_data["timestamp"].dt.strftime('%Y-%m-%dT%H:%M:%S.%f%z')

    events_list = []
    sensor_values = df_sensor_data['sensor_value'].tolist()

    laptop_height_cm = 2.0
    rolling_number_activities = dict.fromkeys(('laptop taken from stack', 'laptop added to stack'), 1)
    prev_value = sensor_values[0]
    for index, row in df_sensor_data.iterrows():
        curr_value = row['sensor_value']
        if curr_value >= prev_value + laptop_height_cm:
            prev_value = curr_value
            curr_activity = "laptop taken from stack"
            rolling_event_id = rolling_number_activities.get(curr_activity)
            curr_event_id = curr_activity + "_" + str(rolling_event_id)
            rolling_number_activities[curr_activity] = rolling_event_id + 1
        elif curr_value <= prev_value - laptop_height_cm:
            prev_value = curr_value
            curr_activity = "laptop added to stack"
            rolling_event_id = rolling_number_activities.get(curr_activity)
            curr_event_id = curr_activity + "_" + str(rolling_event_id)
            rolling_number_activities[curr_activity] = rolling_event_id + 1
        else:
            prev_value = curr_value
            continue
        curr_timestamp = row['timestamp']
        curr_rel_obj = [object_id]
        curr_event_attr = {"event_source": "sensor file " + sensor_file_name}
        events_list.append([curr_event_id, curr_timestamp, curr_activity, curr_rel_obj, curr_event_attr])

    df_events = pd.DataFrame(events_list, columns = ['ocel:eid', 'ocel:timestamp', 'ocel:activity', 'ocel:omap', 'ocel:vmap']) #create final events dataframe
    return df_events


def sensorToOCEL(session_path, sensor_file_path, event_trigger_func_name, object_id, object_class, object_attr):
    """
    sensorToOCEL handles a sensor data file by managing which event trigger function should be used on the given sensor data file.
        It appends all found events and involved objects to the respective event and object file of the current session (sensor_events.pkl and sensor_objects.pkl)
        Customizable: Adjust this function manually by insert the names of the manually created preprocessing and event trigger functions
        in the "if/elif" block in the beginning of this function

    :session_path: The path to the current session folder including a '/' at the end.
    :sensor_file_path: The path to the sensor file as .csv, e.g., "data/sensor_discrete.csv".
    :event_trigger_func_name: The name (as string) of the manually defined event trigger function that is used on the sensor data file.
    :object_id: The object ID that is assigned to the sensor. A sensor can be part of an object (e.g., of a machine) or can be an object itself
    :object_class: The object class of the sensor
    :object_attr: The attributes of the sensor object
    """
    
    sensor_file_name = Path(sensor_file_path).name

    if event_trigger_func_name == "eventTriggerDiscreteStandard":
        df_events = eventTriggerDiscreteStandard(sensor_file_path, object_id, sensor_file_name)
    elif event_trigger_func_name == "eventTriggerContinuousStandard":
        df_events = eventTriggerContinuousStandard(sensor_file_path, object_id, sensor_file_name)
    elif event_trigger_func_name == "eventTriggerCustomAirQualitySensor":
        df_events = eventTriggerCustomAirQualitySensor(sensor_file_path, object_id, sensor_file_name)
    elif event_trigger_func_name == "eventTriggerCustomDistanceSensor":
        df_events = eventTriggerCustomDistanceSensor(sensor_file_path, object_id, sensor_file_name)
    else:
        print("Selected event trigger function not found. Function is aborted")
        return

    #saving events to "sensor_events.pkl" in session folder
    if not os.path.isfile(session_path + "sensor_events.pkl"): #check if "sensor_events.pkl" already exists in session path
        df_events = df_events.sort_values(['ocel:timestamp'], ascending=True, ignore_index=True) #sort by timestamp
        df_events.to_pickle(session_path + "sensor_events.pkl") #create initial sensor_events.pkl file with current df_events
    else:
        updated_events = pd.read_pickle(session_path + "sensor_events.pkl")
        updated_events = pd.concat([updated_events, df_events], ignore_index=True) #append new events to "sensor_events.pkl"
        updated_events = updated_events.sort_values(['ocel:timestamp'], ascending=True, ignore_index=True) #sort by timestamp
        updated_events.to_pickle(session_path + "sensor_events.pkl") #save the updated df_events

    #saving objects to "sensor_objects.pkl" in session folder
    #if there is not yet an object source attribute stating that the object was originally found in the process log or video, add this attribute stating that the source is the sensor
    new_obj_attributes = object_attr
    if object_attr.get("object_source_original") == None:
        new_obj_attributes["object_source_original"] = "sensor file " + sensor_file_name #as a standard attribute save for later that this object initially was found in the sensor file
    df_curr_object = pd.DataFrame([{'ocel:oid': object_id, 'ocel:type': object_class, 'ocel:ovmap': new_obj_attributes}])
    if not os.path.isfile(session_path + "sensor_objects.pkl"): #check if "sensor_objects.pkl" already exists in session path
        df_curr_object.to_pickle(session_path + "sensor_objects.pkl") #create initial sensor_objects.pkl file with one entry containing the current object_id, object_class, and object_attr
    else:
        updated_objects = pd.read_pickle(session_path + "sensor_objects.pkl")
        updated_objects = pd.concat([updated_objects, df_curr_object], ignore_index=True) #append new objects to "sensor_objects.pkl"
        updated_objects.drop_duplicates(subset=['ocel:oid'], keep='last', inplace=True, ignore_index=True) #Keep the last entry since it represents the objec_id with the newest attributes added
        updated_objects.to_pickle(session_path + "sensor_objects.pkl") #save the updated df_objects

def getEventTriggerFuncNames():
    """
    getEventTriggerFuncNames returns a list of all available event trigger functions that can be selected.
        Customizable: Insert the names of the manually created event trigger functions in the list in this function.

    :return: A list containing the names of all available event trigger functions that can be selected.
    """    

    all_event_trigger_function_names = ["eventTriggerDiscreteStandard", "eventTriggerContinuousStandard",
                                        "eventTriggerCustomAirQualitySensor", "eventTriggerCustomDistanceSensor"]
    return all_event_trigger_function_names

if __name__ == "__main__":
  
