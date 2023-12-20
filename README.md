# CS640-UBCOCEAN-Competition

[Competition Link](https://www.kaggle.com/competitions/UBC-OCEAN)

### Installation Instructions
For installation instructions, refer to
[Installation Guide](installation.md)

## Introduction 
Our project involved tackling the UBC OCEAN competition, where we were provided with 70% of the data to build a model. Later, our code was evaluated on a hidden test set comprising 30% of the available data.

### Step 1: Patch Creation using CLAM
[https://github.com/mahmoodlab/CLAM/tree/master](CLAM)

We utilized CLAM for preprocessing and patch extraction. The images we worked with varied in resolution and size compared to the TCGA, LUCC dataset, so we had to experiment with different thresholds

Instead using the presets [ubc_ocean.csv](ubc_ocean.csv)


CLAM pipeline 

<img src="images/preprocessing-pipeline.png" alt="Pre processing " height="300" align="center"/>

OTSU thresholding 

 <img src="images/otsu_thresholding.png" alt="Pre processing " height="300" align="center" />


Final result after Foreground background Seperation

<img src="images/foreground seperation.png" alt="Pre processing " height="300" align="center" />

The green regions represent selected foreground, while the blue regions were removed as holes in the image.

### Step 2: Generating features using pre-trained model
To generate features for all patches, we utilized Transpath alongside pre-trained models. 
[https://github.com/Xiyue-Wang/TransPath](TransPath)

Download weights for ctranspath from the official repo

Execute the code
[extract_features_ctranspath.py](extract_features_ctranspath.py) inside the CLAM environment.

## Step 3:  Building graph using features for patches
Utilize the patch features generated in the previous step to construct graphs.

Execute the script 
[build_graph.py](build_graph.py)

## Step 4: Running models
Run graph based models for multiple instance learning
[train_simple_model.py](train_simple_model.py)


Scores for different models tried:

| Prediction model   | Feature Extraction model | Balanced Accuracy, mean%(std%) |
|--------------------|--------------------------|--------------------------------|
| CLAM               | Resnet50                 | 65.2 (5.2)                     |
| CLAM               | cTranspath               | 74.09 (7.3)                    |
| Graph Transformer  | Resnet18                 | 68.5 (3.8)                     |
| Graph Transformer  | cTranspath               | **77.3** (5.7)                 |


## Conclusion
The final model comprised an ensemble of models generated in each fold within the Graph Transformer architecture.
Our model got balanced accuracy of 75%