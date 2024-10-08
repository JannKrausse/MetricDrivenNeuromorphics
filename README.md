# MetricDrivenNeuromorphics
This is the repository with the code and supplementary information of our paper "On Metric-Driven Development of Embedded Neuromorphic AI" at IEEE SOCC'24.

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

If you use our work please cite it using the following citation:
```bibtex
@inproceedings{krausse202metric,
  title={On Metric-Driven Development of Embedded Neuromorphic AI},
  author={Krausse, Jann and Neher, Moritz and Fuerst-Walter, Iris and Weigelt, Carmen and Harbaum, Tanja and Knobloch, Klaus and Becker, Juergen},
  booktitle={2024 IEEE 37th International System-on-Chip Conference (SOCC)},
  year={2024},
  organization={IEEE}
}
```

Lastly, it should be noted that the code contains some parts of the open source neuromorphic framwork Norse (https://github.com/norse/norse). It is written in the files whereever that is the case.
