# FilletRec

Code for FilletRec: A Lightweight Graph Neural Network with Intrinsic Features for Automated Fillet Recognition.

![The overall pipeline of FilletRec](img/overview.jpg)

## About FilletRec

Automated recognition and simplification of fillet features in CAD models is critical for CAE analysis, yet it remains an open challenge. Traditional rule-based methods lack robustness, while existing deep learning models suffer from poor generalization and accuracy on complex fillets due to their generic design and inadequate training data. To address these issues, this paper proposes an end-to-end, data-driven framework specifically for fillet features. We first construct and release a large-scale, diverse benchmark dataset for fillet recognition to address the inadequacy of existing data. Based on it, we propose FilletRec, a lightweight graph neural network. The core innovation of this network is its use of pose-invariant intrinsic geometric features, such as curvature, enabling it to learn more fundamental geometric patterns and thereby achieve high-precision recognition on complex geometric topologies. Experiments show that FilletRec surpasses state-of-the-art methods in both accuracy and generalization, while using only 0.2\%-5.4\% of the parameters of baseline models, demonstrating remarkable model efficiency. Finally, the framework completes the automated workflow from recognition to simplification by integrating a novel geometric simplification algorithm.

## Preparation

### Requirements
- python >= 3.8
- tensorflow >= 2.13.0
- pythonocc-core >= 7.5.1 (more info here: https://github.com/tpaviot/pythonocc-core.git)
- occwl (more info here: https://github.com/AutodeskAILab/occwl.git)
- numpy
- scikit-learn

### Environment setup

```
git clone https://github.com/Miss-Hedgehog/FilletRec.git
cd FilletRec
conda env create -f environment.yml
conda activate filletrec
```
If the environment.yml installation fails, you can install the required packages manually.
### Data preparation

Our synthetic Fillet datasets have been publicly available on [Science Data Bank](https://www.scidb.cn/en/detail?dataSetId=931c088fd44f4d3e82891a5180f10d90)

## Usage

### 1.Training

For fillet recognition, this network can be trained using:

```
python ./train.py
```

The best checkpoint based on the smallest validation loss will be stored in a folder called `checkpoint`. 

### 2.Testing

```
python ./test.py
```

### 3.Predicting

```
python ./test_and_save.py
```

### 4.Visualizing

```
python ./visualizer.py
```

### 5.Removing


#### 5.1 Expanding

```
python ./simplify_fillet_mesh.py
```

#### 5.2 Intersecting


#### 5.3 Cleaning

```
python ./delete_boundary_tri.py
```


