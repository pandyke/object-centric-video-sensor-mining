# Unstructured Data in Object-Centric Process Mining

This repository contains the source code for the paper 'Refining the Process Picture: Unstructured Data in Object-Centric Process Mining'. 
For a more comprehensive overview of the approach we refer to the paper.


## Installation

Execute the following commands to check out the repository, and create a new Python virtual environment.

```
git clone https://github.com/550e8400e29b41d4a716446655440000/object-centric-video-sensor-mining
python3 -m venv venv
source venv/bin/activate
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
- Start: 
- Process Log: 
- Video Object Labeler: 
- Video Object Area Definer: 
- Video Event Extractor: 
- Sensor: 
- Analysis: 


The following screenshot shows the object labeling step of the prototype used on the Solve4X dataset ([Chvirova et al. 2024](https://doi.org/10.1016/j.dib.2024.110716)).
![Alt text](https://github.com/550e8400e29b41d4a716446655440000/object-centric-video-sensor-mining/blob/main/figures/OCVSM_prototype_screenshot.PNG?raw=true "Screenshot")
