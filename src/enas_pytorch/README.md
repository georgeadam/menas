# Efficient Neural Architecture Search (ENAS) in PyTorch

PyTorch implementation of [Efficient Neural Architecture Search via Parameters Sharing](https://arxiv.org/abs/1802.03268).

<p align="center"><img src="assets/ENAS_rnn.png" alt="ENAS_rnn" width="60%"></p>

**ENAS** reduce the computational requirement (GPU-hours) of [Neural Architecture Search](https://arxiv.org/abs/1611.01578) (**NAS**) by 1000x via parameter sharing between models that are subgraphs within a large computational graph. SOTA on `Penn Treebank` language modeling.

**\*\*[Caveat] Use official code from the authors: [link](https://github.com/melodyguan/enas)\*\***


## Prerequisites

- Python 3.6+
- [PyTorch](http://pytorch.org/)
- tqdm, scipy, imageio, graphviz, tensorboardX

## Usage

Install prerequisites with:

    conda install graphviz
    pip install -r requirements.txt
    
### RNN Cell Search

#### ENAS

To train **ENAS** to discover a recurrent cell for RNN:

    python train_scripts/train_regular.py --network_type rnn --dataset ptb --controller_optim adam \
                                          --controller_lr 0.00035 --shared_optim sgd --shared_lr 20.0 \ 
                                          --entropy_coeff 0.0001 --train_type orig --mode train

    python train_scripts/train_regular.py --network_type rnn --dataset wikitext --train_type orig
    
    
#### ENAS - Flexible

To train **ENAS** where the controller dynamically chooses how many nodes should be in every sampled DAG as opposed
to having this be something that the user specifies beforehand run the following command:

    python train_scripts/train_regular.py --network_type rnn --dataset ptb --controller_optim adam \
                                          --controller_lr 0.00035 --shared_optim sgd --shared_lr 20.0 \ 
                                          --entropy_coeff 0.0001 --train_type flexible --mode train \
                                          --num_blocks 12
                                          
Note that `--num_blocks` now dictates the *maximum* number of nodes that a sampled DAG can have. Keep this greater
than or equal to 3. Current code overwrites any number sampled less than 3 with 3, though that might ruin the gradient 
signal somehow (needs checking).

    
#### Hardcoded Architectures

To train **ENAS** to figure out just the activation function combination for a given hardcoded architecture:

    python train_scripts/train_regular.py --network_type rnn --dataset ptb --controller_optim adam \
                                          --controller_lr 0.00035 --shared_optim sgd --shared_lr 20.0 \ 
                                          --entropy_coeff 0.0001 --train_type hardcoded --architecture tree \
                                          --mode train
                                        
    python train_scripts/train_regular.py --network_type rnn --dataset ptb --controller_optim adam \
                                          --controller_lr 0.00035 --shared_optim sgd --shared_lr 20.0 \ 
                                          --entropy_coeff 0.0001 --train_type hardcoded --architecture chain \
                                          --mode train   
    
#### Random Architectures

To train shared parameters by sampling random architectures instead of using a controller:

    python train_scripts/train_regular.py --network_type rnn --dataset ptb --controller_optim adam \
                                          --controller_lr 0.00035 --shared_optim sgd --shared_lr 20.0 \ 
                                          --entropy_coeff 0.0001 --train_type random --mode train
                                              
#### Training From Scratch

To train from scratch according to the following procedure

* Load trained controller and shared parameters, *C* and *S* respectively from dir specified by `load_path`
* (Note that you need to specify the correct `train_type` that was used to train *C* and *S*. This will be in `load_path`
name like `ptb_{train_type}_{datetime}`.
* Sample a decent architecture *A* via the `derive()` method using *C* and *S* 
* Reset the shared parameters to random initialization and train them from scratch for the sampled architecture. 
Controller no longer used

run the following command

    python train_scripts/train_regular.py --network_type rnn --dataset ptb --train_type orig 
                                          --mode train_scratch
                                          
Specify different parameters such as `--max_epoch x` to train for more or less epochs than were used to 
train ENAS.

### Convolutional Cell Search (in progress)

To train **ENAS** to discover CNN architecture:

    python main.py --network_type cnn --dataset cifar --controller_optim momentum --controller_lr_cosine=True \
                   --controller_lr_max 0.05 --controller_lr_min 0.0001 --entropy_coeff 0.1

or you can use your own dataset by placing images like:

    data
    ├── YOUR_TEXT_DATASET
    │   ├── test.txt
    │   ├── train.txt
    │   └── valid.txt
    ├── YOUR_IMAGE_DATASET
    │   ├── test
    │   │   ├── xxx.jpg (name doesn't matter)
    │   │   ├── yyy.jpg (name doesn't matter)
    │   │   └── ...
    │   ├── train
    │   │   ├── xxx.jpg
    │   │   └── ...
    │   └── valid
    │       ├── xxx.jpg
    │       └── ...
    ├── image.py
    └── text.py

To generate `gif` image of generated samples:

    python generate_gif.py --model_name=ptb_2018-02-15_11-20-02 --output=sample.gif

More configurations can be found [here](configs/config_ours.py).

## Ablation Studies

Results of ablation studies will be stored in the `analysis_results` dir which will be automatically created if it 
does not yet exist. There, results are further separated into different subdirectories based on the ablation study
name.

### Activation Function Replacement

To see the effect of replacing all activation functions with a single type for all possible types of activation 
functions:

    python ablation_studies/activation_replacement.py --load_path=ptb_2018-02-15_11-20-02 --mode=derive
    
### Node Removal

To see the effect of removing single nodes from a DAG *G* sampled via the `derive()` method for an already trained set
of controller and shared parameters, *C* and *S* respectively that were saved in the log directory given by 
`--load_path`, run the 
following command

    python ablation_studies/node_removal.py --load_path=ptb_orig_2018-11-25_17-22-36 --mode=derive


## Results

Efficient Neural Architecture Search (**ENAS**) is composed of two sets of learnable parameters, controller LSTM *θ* and the shared parameters *ω*. These two parameters are alternatively trained and only trained controller is used to derive novel architectures.

### 1. Discovering Recurrent Cells

![rnn](./assets/rnn.png)

Controller LSTM decide 1) what activation function to use and 2) which previous node to connect.

The RNN cell **ENAS** discovered for `Penn Treebank` and `WikiText-2` dataset:

<img src="assets/ptb.gif" alt="ptb" width="45%"> <img src="assets/wikitext.gif" alt="wikitext" width="45%">

Best discovered ENAS cell for `Penn Treebank` at epoch 27:

<img src="assets/best_rnn_epoch27.png" alt="ptb" width="30%">

You can see the details of training (e.g. `reward`, `entropy`, `loss`) with:

    tensorboard --logdir=logs --port=6006


### 2. Discovering Convolutional Neural Networks

![cnn](./assets/cnn.png)

Controller LSTM samples 1) what computation operation to use and 2) which previous node to connect.

The CNN network **ENAS** discovered for `CIFAR-10` dataset:

(in progress)


### 3. Designing Convolutional Cells

(in progress)


## Reference

- [Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578)
- [Neural Optimizer Search with Reinforcement Learning](https://arxiv.org/abs/1709.07417)


## Author

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
