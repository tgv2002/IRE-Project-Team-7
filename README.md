# IRE-Project-Team-7
Entire implementation of baselines, final architecture and application - 'Multi-class classification of Mental Health Disorders', as a part of major project of IRE course (team 7).

## Instructions for using the system

* Clone this repository into your local system.
* Download the `models` folder from [here](https://drive.google.com/drive/folders/1VU1JTBm-D7GCQ1oWOa1Srwq90yw4eLma) and move it into the folder of this repository (Google drive zips the whole folder into another folder on downloading, make sure you move only the folder named `models` to this folder).
* Execute the `run.sh` script in the folder via the command `./run.sh` after giving it permission for execution via the command `chmod 777 run.sh`. This script sets up a new virtual environment with python3.8, installs cuda toolkit, pytorch, transformers, streamlit libraries, and moves into the app folder. Feel free to modify the script for use, as per requirements. 
* After above script completes necessary installations and setup and moves you into the app folder, execute the app locally by running the command: `streamlit run main.py`. Open the URL provided by streamlit (or wait till it opens automatically on your browser).
* Follow the instructions on the dashboard, and enter the text for classification in the text area, and submit it when done. The prediction of the model is displayed on submitting.
