# Urban vehicle trajectory learning via transformers

### Requirements
In addition to the STAR and PNEUMA libraries, the following libraries are required:

```bash
pip install einops==0.6.0
pip install matplotlib==3.6.1
pip install wandb==0.13.4
```

### Training and Testing

This repository contains code for training and evaluating a deep learning model to predict the future trajectory of vehicles. The default settings for training are a sequence length of 10, 400 epochs, and a neighborhood of 100.

To train the original model with pedestrian datasets, simply keep the 'output/eth' folder as it is and use the standard commands.


### Model Versions

The 'star.py' file is the original version of the model, which uses normalized batches of trajectories and does not include a map in the input features. Alternative versions of the model that use absolute trajectories in spatial or temporal encoding, or both, or that use absolute spatial encoding plus map features, are available in the 'extra_files/star_files' directory.


### Preprocessing Vehicle Datasets

The 'dataprocess.ipynb' notebook allows you to create new pickle files of trajectory batches using the Pneuma dataset, as well as new map files. You can specify the input file and desired parameters, and the resulting files will be saved in the 'output' folder. Precomputed dataset CSV files can be found in the 'data_pneuma' folder.


### Ready-to-use Files

To train and evaluate the model, run the 'trainval.py' file. Pre-trained models are available in the 'extra_files/saved_models' directory, and can be tested by creating an 'output/eth/star' folder and copying the desired model files into it.

Alternatively, you can use pickle files in the 'extra_files/cpkl_files' directory for training and evaluation. The '0900_0930_train' file can be used as the training batch, and a file between '0900_0930_train' and '0900_0930_train' can be used as the validation batch. To do this, copy the two files into the 'output/eth' folder and run the 'trainval.py' script. The remaining pickle file can then be used for testing by replacing the 'test_batch_cache.cpkl' file previously copied.


### Plotting with wandb.com

By default, the script uses wandb.com to plot results and metrics for both training and testing. You can set the 'wandb_log' as False in the 'trainval.py' script to disable this behavior. During testing, you can choose to plot normalized or absolute trajectories and plot the neighbors in the scene by changing the 'absolutePlot' and 'neigPlot' options.


### Additional Resources

See also README_pneuma.rst and README_star.md files to have a better understanding of how to install required packages and run the files present in the project folder.


### Reference

The code base heavily borrows from [STAR](https://github.com/Majiker/STAR)
