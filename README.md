# Unstructured Data in Object-Centric Process Mining

This repository contains the source code for the paper 'Refining the Process Picture: Unstructured Data in Object-Centric Process Mining'. 
For a more comprehensive overview of the approach we refer to the ([paper](https://www.sciencedirect.com/science/article/pii/S0306437925000663#:~:text=To%20answer%20this%20research%20question%2C%20we%20propose%20the,and%20traditional%20event%20logs%20for%20object-centric%20process%20mining)).


## Installation

Execute the following commands to check out the repository, create a new Python virtual environment, and install the dependencies.

```
git clone https://github.com/pandyke/object-centric-video-sensor-mining
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Start the graphical user interface (file `gui.py`) with the following command.

```
python3 gui.py
```

## Functionality: Object-Centric Video and Sensor Mining

The file `gui.py` starts the graphical user interface of the software prototype and guides the user through the individual steps.
The prototype guides the user through the steps required for combining structured process data with unstructured video and sensor data for object-centric process mining.
Thereby, the user functions as a domain expert and can give input at specific steps.
The output is a valid OCEL 2.0 file comprising events and objects that were extracted from the various data sources.

Thereby, the menu shows the following steps with respective functionalities:
- Start: Allows starting a new session or loading an existing session. By starting a new session a folder is created that saves all your work already done with all respective files. You can then later load this session and continue where you stopped.
- Process Log: A process log can be loaded, either already in the valid OCEL format or from a file in table format that is then preprocessed.
- Video Object Labeler: In this step, a video file can be selected on which an object tracking algorithm is then run. The results of the algorithm are recognized objects with bounding boxes which can then be manually labelled.
- Video Object Area Definer: Here, own static objects can be manually defined in the video frame. If the object tracking algorithm does not recognize certain objects but these are important for the process (e.g., a cupboard from which items are regularly removed or placed in), one can define these objects here by drawing manual bounding boxes in the video frame and labelling them.
- Video Event Extractor: This step allows for defining event rules in the form of: When the bounding boxes of OBJECT1 and OBJECT2 start overlapping an event called ACTIVITYNAME is created. Thereby, OBJECT1, OBJECT2, and ACTIVITYNAME can be defined and several rules for different objects can be set.
- Sensor: Here, sensor CSV files containing timestamps and sensor values can be loaded (discrete and continuous sensor data are both posssible). The sensor can then be allocated to a known object or a new object can be defined for the sensor. Moreover, rules can be defined as to when events should be created from the sensor values. For continuous data, a rule is in the form: When the sensor values in a certain timeframe change greater or smaller than a threshold then an event with a certain name is created. For discrete sensor values, in the form: When the sensor switches to a certain state, then an event with a certain name is created.
- Analysis: The last step creates a valid OCEL file based on all the previous steps, then shows basic statistics on this file (e.g., number of events and objects) and finally saves and shows directly follows multigraphs of the OCEL file, displaying all involved objects and events.


The following screenshot shows the object labeling step of the prototype used on the Solve4X dataset ([Chvirova et al. 2024](https://doi.org/10.1016/j.dib.2024.110716)).
![Alt text](https://github.com/550e8400e29b41d4a716446655440000/object-centric-video-sensor-mining/blob/main/figures/OCVSM_prototype_screenshot.PNG?raw=true "Screenshot")
