import pm4py
from pm4py.objects.ocel.validation import jsonocel
import json
import pandas as pd
from datetime import datetime

def analyzeOCEL(ocel, method, view=True, save=False, savepath=None):
    """
    analyzeOCEL analyzes an ocel log with one of the predefined methods and can save the resulting .svg.

    :ocel: The ocel to analyze
    :method: One of the predefined analysis methods/visualisations as string
    :view: Boolean whether or not the results should be visualized on the screen
    :save: Boolean whether or not the results should be saved as .svg
    :savepath: If save=True, the path as string to the directory where the results should be saved, including an "/" at the end (e.g., "results/")
    """
    
    if method == "dfg_frequency":
        #views the model with the frequency annotation
        ocdfg = pm4py.discover_ocdfg(ocel)
        if view:
            pm4py.view_ocdfg(ocdfg, format="svg", annotation="frequency", bgcolor="white")
        if save:
            pm4py.save_vis_ocdfg(ocdfg, file_path=savepath + "dfg_freq_ocel.svg", annotation='frequency', bgcolor="white")
    elif method == "dfg_performance":
        #views the model with the performance annotation
        ocdfg = pm4py.discover_ocdfg(ocel)
        if view:
            pm4py.view_ocdfg(ocdfg, format="svg", annotation="performance", performance_aggregation="median", bgcolor="white")
        if save:
            pm4py.save_vis_ocdfg(ocdfg, file_path=savepath + "dfg_perf_ocel.svg", annotation='performance', performance_aggregation="median", bgcolor="white")
    elif method == "petri_net":
        #object-centric petri-net
        model = pm4py.discover_oc_petri_net(ocel)
        if view:
            pm4py.view_ocpn(model, format="svg", bgcolor="white")
        if save:
            pm4py.save_vis_ocpn(model, file_path=savepath + "ocpn_ocel.svg", bgcolor="white")
    else:
        print("no valid analysis method selected")

def flattenOCEL(ocel, objectType, save=False, savepath=None):
    """
    flattenOCEL converts an object-centric event log to a traditional event log with the specification of an object type and saves it if needed

    :ocel: The ocel to flatten
    :objectType: The object type on which the ocel should be flattened on
    :save: Boolean whether or not the traditional event log should be saved
    :savepath: If save=True, the path as string to the directory where the results should be saved, including an "/" at the end (e.g., "results/")
    :return: A flattened traditional event log
    """

    #Check if object type is in log
    all_object_types = pm4py.ocel_get_object_types(ocel)
    if objectType not in all_object_types:
        print("The specified object type is not in the provided ocel")
        #print(all_object_types)
        return None
    
    flatLog = pm4py.ocel_flattening(ocel, objectType)
    if save:
        pm4py.write_xes(flatLog, savepath + 'flattened_log_' + objectType + '.xes')
    
    return flatLog

def analyzeXES(xes_log, method, view=True, save=False, savepath=None):
    """
    analyzeXES analyzes a traditional .xes log with one of the predefined methods and can save the results.

    :xes_log: The .xes log to analyze
    :method: One of the predefined analysis methods/visualisations as string
    :view: Boolean whether or not the results should be visualized on the screen
    :save: Boolean whether or not the results should be saved
    :savepath: If save=True, the path as string to the directory where the results should be saved, including an "/" at the end (e.g., "results/")
    """
    
    if method == "dfg":
        #views the directly-follows graph
        dfg, start_activities, end_activities = pm4py.discover_dfg(xes_log)
        if view:
            pm4py.view_dfg(dfg, start_activities, end_activities, format='svg', bgcolor="white")
        if save:
            pm4py.save_vis_dfg(dfg, start_activities, end_activities, file_path=savepath + "dfg_xes.svg", bgcolor="white")
    elif method == "dfg_performance":
        #views the directly-follows graph with performance
        performance_dfg, start_activities, end_activities = pm4py.discover_performance_dfg(xes_log)
        if view:
            pm4py.view_performance_dfg(performance_dfg, start_activities, end_activities, format='svg', bgcolor="white")
        if save:
            pm4py.save_vis_performance_dfg(performance_dfg, start_activities, end_activities, file_path=savepath + "dfg_perf_xes.svg", bgcolor="white")
    elif method == "alpha_miner":
        #views the alpha miner petri net
        net, initial_marking, final_marking = pm4py.discover_petri_net_alpha(xes_log)
        if view:
            pm4py.view_petri_net(net, initial_marking, final_marking, format='svg', bgcolor="white")
        if save:
            pm4py.save_vis_petri_net(net, initial_marking, final_marking, file_path=savepath + "alpha_xes.svg", bgcolor="white")
    else:
        print("no valid analysis method selected")

def dataframeToOCEL(df_objects, df_events):
    """
    dataframeToOCEL creates an ocel log from two dataframes containing the objects and events with relations

    :df_objects: dataframe containing all objects with their attributes. Need the columns 'ocel:oid', 'ocel:type', and 'ocel:ovmap'
        The 'ocel:ovmap' column contains values in json format (e.g., {"name": "John"}, {"name": "Peter"})
    :df_events: dataframe containing all events with their attributes and relations to objects. Need the columns 'ocel:eid', 'ocel:timestamp', 'ocel:activity', 'ocel:omap', and 'ocel:vmap'
        The 'ocel:vmap' column contains values in json format (e.g., {"Attr1": "AttrValue1"}, {"Attr2": "AttrValue2"})
        The 'ocel:omap' column contains values in a list like format (e.g., ["object1", "object2"], ["object1", "object3"])
    :return: Returns a json string serialization with the created ocel
    """

    #Get the ocel template
    with open('data/ocel_template.jsonocel') as f:
        ocel_template = json.load(f)

    #Add object types
    all_object_types = df_objects['ocel:type'].unique()
    ocel_template["ocel:global-log"]["ocel:object-types"] = list(all_object_types)

    #Add objects
    df_objects.set_index('ocel:oid', inplace=True)
    objects_json = df_objects.to_json(orient="index", indent=2)
    ocel_template["ocel:objects"] = json.loads(objects_json)

    #Add events
    df_events.set_index('ocel:eid', inplace=True)
    df_events['ocel:timestamp'] = df_events['ocel:timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S.%f%z')
    events_json = df_events.to_json(orient="index", indent=2)
    ocel_template["ocel:events"] = json.loads(events_json)

    return ocel_template

def get_ocel_from_ocel1_or_ocel2(path_to_ocel):
    """
    get_ocel_from_ocel1_or_ocel2 checks if a file is a valid ocel1 or ocel2 file and returns an ocel object in these cases. Returns None otherwise.

    :path_to_ocel: The path to the ocel file.
    :return: Returns an ocel file or None
    """
    if jsonocel.apply(path_to_ocel, "data/ocel2_schema.json"):
        print("Is valid OCEL2 format")
        ocel = pm4py.read_ocel2_json(path_to_ocel)
        return ocel
    elif jsonocel.apply(path_to_ocel, "data/ocel1_schema.json"):
        print("Is valid OCEL1 format")
        if path_to_ocel.endswith("jsonocel"):
            ocel = pm4py.read_ocel(path_to_ocel)
            return ocel
        else:
            print("Valid OCEL1 file has to end with .jsonocel")
            return None
    else:
        print("File is not a valid OCEL1 or OCEL2 file. Please check format against the 'ocel1_schema.json' or 'ocel2_schema.json' file. Function aborted")
        return None
    
def replace_null_qualifier_ocel2(path_to_ocel2):
    """
    replace_null_qualifier_ocel2 takes the path to an ocel2 file and checks if some qualifiers in relationships are null and replaces them with ""

    :path_to_ocel2: The path to the ocel2 file.
    """
    with open(path_to_ocel2, "r") as ocel2_json_in:
        data = json.load(ocel2_json_in)

    all_events = data["events"]
    for event in all_events:
        all_relationships = event["relationships"]
        for relationship in all_relationships:
            if relationship["qualifier"] == None:
                relationship["qualifier"] = ""
    
    with open(path_to_ocel2, "w") as ocel2_json_out:
        json.dump(data, ocel2_json_out)


if __name__ == "__main__":
    
