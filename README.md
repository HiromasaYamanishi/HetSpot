## Overview
Code for "HetSpot: Analyzing Tourist Spot Popularity with Heterogeneous Graph Neural Network" @IVSP24
<a href="https://dl.acm.org/doi/abs/10.1145/3655755.3655770" target="_blank">[paper]</a>
## enviroment
```bash
pip install -r requirements.txt
```
data can be down loaded from <a href="https://drive.google.com/drive/folders/1NrXtCZRyXtfZ3xkYzSPTfHdZSZPJ1W1n?usp=drive_link" target="_blank">[data]</a>

place under root directory and
```bash
unzip data.zip
```
## Usage
```python
python experiment.py --word --city --pref --category --model_type lstm
```
