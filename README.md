# [Graph-Attention Augmented Temporal Neural Network Build for Diagnosis](https://www.topcoder.com/challenges/c8019040-9be1-4a2c-bbf1-0007c39cf0e6?tab=winners)

## Task 
The task of this challenge is to read the paper and independently implement GATE, using the dataset given in this challenge.

## Tech Stack
- [Anaconda Python3](https://www.anaconda.com/distribution/)
- [Pytorch](https://pytorch.org/)

## Setup
```CONDA
> conda create -n gate python==3.7 -y
> conda activate gate
> pip install jupyter==1.0.0 networkx==2.4 pandas==1.1.1 numpy==1.20.2 matplotlib==3.0.3 tqdm==4.40.0  scikit-learn==0.22.1 decorator==4.4.1 ipywidgets
> install pytorch following below instructions
> jupyter nbextension enable --py widgetsnbextension
> cd path/to/project/submission/
> run command "jupyter notebook"
> open notebook and run 
```
#### Installing pytorch 
```CONDA
windows without cuda - pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
linux without cuda - pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
mac without cuda - pip install torch torchvision
```

#### Please Read Through the markdowns in jupyter notebooks to re-run.
#### Just for evalaution that the script is working one should use a smaller no of epochs like  2-3 should work fine.