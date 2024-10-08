# MetricDrivenNeuromorphics
This is the repository with the code and supplementary information of our paper "On Metric-Driven Development of Embedded Neuromorphic AI" at IEEE SOCC'24. In one sentence, the paper presents an approach that automizes neural network training given a set of requirements for a specific system (e.g., IoT or Safety-Critical). Although it assesses neuromorphic networks, the approach can also well be applied to any type of neural network.

## Installation and Running Experiments
The requirements should be installed in a Python 3.9 environment. 
```bash
pip install -r requirements.txt
```
For execution, run either 
```bash
python3 main.py
```
for a single training or
```bash
python3 optuna_hps.py
```
for a whole metric-centered optimization run. Be sure to set the respective desired flags.

Lastly, it should be noted that the code contains some parts of the open source neuromorphic framwork Norse (https://github.com/norse/norse). It is written in the files whereever that is the case.

## Citation
If you use our work, please consider citing it using the below citation.
```bibtex
@inproceedings{krausse202metric,
  title={On Metric-Driven Development of Embedded Neuromorphic AI},
  author={Krausse, Jann and Neher, Moritz and Fuerst-Walter, Iris and Weigelt, Carmen and Harbaum, Tanja and Knobloch, Klaus and Becker, Juergen},
  booktitle={2024 IEEE 37th International System-on-Chip Conference (SOCC)},
  year={2024},
  organization={IEEE}
}
```
