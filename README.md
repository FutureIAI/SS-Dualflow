# SS-Dualflow

## Installation

```bash
pip install -r requirements.txt
```

## Data
We use [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) to verify the performance.

The dataset is organized in the following structure:
```
mvtec-ad
|- bottle
|  |- train
|  |- test
|  |- ground_truth
|- cable
|  |- train
|  |- test
|  |- ground_truth
...
```
## Train and eval
ResNet18 
```bash
# train
python main.py -cfg configs/resnet18.yaml --data path/to/mvtec-ad -cat [category]

# eval
python main.py -cfg configs/resnet18.yaml --data path/to/mvtec-ad -cat [category]  
#-cfg configs/resnet18.yaml --data /home/cgz/MZL/dateset/mvtec-ad -cat zipper 
```

