# [Autoencoder](https://github.com/krayush07/autoencoder/)
This repository contains code for feature dimensionality reduction using autoencoder. The repository will be updated with other methods to encode the input as well as code to train autoencoder on textual dataset.

<br/>

# Requirements
* Python 2.7<br/>
* Tensorflow 1.2.1<br/>
* Numpy<br/>

<br/>

# Project Module
* **_[utility_dir](/global_module/utility_dir):_** storage module for data, vocab files, saved models, tensorboard logs, outputs.

* **_[implementation_module](/global_module/implementation_module):_** code for model architecture, data reader, training pipeline and test pipeline.

* **_[settings_module](/global_module/settings_module)_**: code to set directory paths (data path, vocab path, model path etc.), set model parameters (hidden dim, attention dim, regularization, dropout etc.), set vocab dictionary.

* **_[run_module](/global_module/run_module):_** wrapper code to execute end-to-end train and test pipeline.

* **_[visualization_module](/global_module/visualization_module):_** code to generate embedding visualization via tensorboard.

* **_[utility_code](/global_module/utility_code):_** other utility codes

<br/>

# How to run
* **train:** `python -m global_module.run_module.run_train`

* **test:** `python -m global_module.run_module.run_test`

* **visualize tensorboard:** `tensorboard --logdir=PATH-TO-LOG-DIR`

* **visualize embeddings:** `tensorboard --logdir=PATH-TO-LOG-DIR/EMB_VIZ`

<br/>

# How to change model parameters

Go to `set_params.py` [here](/global_module/settings_module/set_params.py).


<br/>


# Loss and Accuracy Plots

![alt text](global_module/utility_dir/mnist/viz/train_loss.png?raw=true "train_loss")

![alt text](global_module/utility_dir/mnist/viz/valid_loss.png?raw=true "valid_loss")

 <br/>
 <br/>


# Autoencoder Visualization

Decoded and corresponding input image of training set at different steps:

![alt text](global_module/utility_dir/mnist/viz/train_3065_3.png?raw=true "'3' at step 3065")

![alt text](global_module/utility_dir/mnist/viz/train_3065_6.png?raw=true "'6' at step 3065")

![alt text](global_module/utility_dir/mnist/viz/train_7390_0.png?raw=true "'0' at step 7390")

![alt text](global_module/utility_dir/mnist/viz/train_7390_2.png?raw=true "'2' at step 7390")

![alt text](global_module/utility_dir/mnist/viz/train_48000_3.png?raw=true "'0' at step 48000")


<br/>


Decoded and corresponding input image of valid set:

![alt text](global_module/utility_dir/mnist/viz/valid_12000_9.png?raw=true "'9' at step 12000")

<br/>

Decoded and corresponding input image of test set:

![alt text](global_module/utility_dir/mnist/viz/test_7.png?raw=true "'7'")

<br/>
<br/>

# Latent Representation

t-SNE representation of latent features of images in test set

![alt text](global_module/utility_dir/mnist/viz/tSNE_test.png?raw=true "'t-SNE of latent features of images in test set. Each (label) digit is represented by a different color'")

<br/>

Visualization of points corresponding to image '1' in t-SNE representation
![alt text](global_module/utility_dir/mnist/viz/label1.png?raw=true "'Points representing images of 1 in test set.'")


<br/>

Visualization of points corresponding to image '5' in t-SNE representation
![alt text](global_module/utility_dir/mnist/viz/label5.png?raw=true "'Points representing images of 5 in test set.'")


