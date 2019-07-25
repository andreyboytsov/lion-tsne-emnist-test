#!/bin/bash -l
#OAR -n job $1
#OAR -l nodes=1/core=1,walltime=1

source ../collectedForHPC/lion-environment/bin/activate
echo job $1 started
python lion_extended_accuracy_plot_data.py $1
