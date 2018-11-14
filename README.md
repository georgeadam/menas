MENAS
==============================

More Efficient Neural Architecture Search.
One Paragraph of project description goes here.

## TODO

### October 29 Meeting
From Meeting:

1. Reproduce existing results on RNN search
    * Just need to be in the ballpark of what they got in the paper
    * Debug to see how long the sequences are for the PTB task: there is a comment in the code saying that backprop 
    through time is truncated at 35 time steps
    * Figure out if how the hidden state for the shared parameters is being stored makes sense 

2. Add this extra gradient term (without actually differentiating through it)
    * Compare this against regular training
    * There are a couple of choices to be made in terms of how we actually get this additional term: momentum stored 
    in the optimizer (least noisy), most recent gradient (most noisy)
    
3. Change the search space such that it is possible to sample LSTM style cells with hadamard products
	* Their DAG is too simplified since for each node in a cell, we just sample an activation function and a previous node to connect it to
    * We should include more complicated connection patterns as an available choice
    
4. We can restrict the choices for activation function to be for the entire cell rather than for each node in the cell
    * Based on some of the cells they came up with, there is alternation between tanh and relu activation for the 
    nodes inside the cell that seems totally arbitrary and likely reduces interpretability of the features created
    
    
### November 12 Meeting

1. Incorporating extra gradient term into ENAS
    * DARTS code has it already, just need to change this to affect controller parameters for ENAS instead
    * Do empirical analysis to see similarity between fake unrolled gradient at step t, and the true gradient at t+1.
     If this similarity is low, then extra gradient should have no benefit

2. Retrain single architecture to get results like in ENAS paper
    * They probably trained a decent controller, sampled like 100 DAGs from it, found the best DAG out of those 100, 
    and then retrained that single DAG from scratch or from the state of the shared parameters at that time
    * Alternatively, we could play around with the entropy term turning it into a penalty once we want to start 
    fine-tuning an architecture. This might require less coded than the first method, though we would have to still 
    keep training the controller parameters in this case, otherwise how else would we decrease the entropy if not by 
    tuning the controller's parameters to output more confident probabilities at each step?

3. Ablation Studies
    * Now that we have the code running for the two ablation studies, let's see what effect they have on the various 
    types of models we've trained (ENAS, random, hardcoded)
    * Create another ablation study that adds connections instead of removing them
    * We want to isolate the most important factors contributing to ENAS' performance. ls it the connection patterns 
    or activation function mixing?
    
4. Architecture design instead of blind search, and model capacity
    * It would be useful to generate architectures with desired properties such as robustness to input 
    perturbations, or to control for model capacity (i.e. have the controller's hidden state be in an embedding space
    where different directions correspond to different model properties like in proper generative models)
    * Need a notion of model capacity for the case where we have a fixe number of nodes and connections. Could be 
    number of unique paths from input to output node or something like that. 
    * What role does mixing activation functions play in model capacity? I.e. does having a mixture of activation 
    functions increase ability to overfit compared to just a single activation type?
    * Do analysis of the controller's findal hidden state after sampling an architecture to see if architectures that
    differ by just a single connection or activation function have similar hidden states in the controller 

5. Random vs Trained Controllers
    * Train something like 100 sampled architectures from a randomly initialized controller individually until 
    convergence, and also train 100 sampled architectures from a converged controller individually until convergence 
    as well
    * Create density plots of performance for these two groups of models
    * Run statistical tests to see if differences between the groups are significant
    * The point is to see if ENAS or DARTS actually provide a benefit, or if it's their limited search space that 
    makes the  architecture search work
    
6. Extending Architecture Search Space
    * To have more flexible model capacity, the controller should be able to choose the hidden state size of the 
    shared nodes. Since parameters are shared, we can just take a block from each weight matrix (i.e. if the size of 
    the weight matrices is 1000x1000 and the controller specifies a hidden state size of 200, then just take a 
    200x200 chunk starting from the top left of the 1000x1000 matrices.)
    * The controller should also be able to specify the number of nodes in a cell. For a given iteration, we might? 
    have to sample DAGs of the same number of nodes in order for this to be compatible with the current setup
    
* Write up missing term and background, etc.


* Fill out the TODO in-depth.
* Fill out the README.
* Construct a skeleton of the final report from the proposal.
* Get ENAS working and write down notes for deployment.
* Make a super simple ENAS, with maybe 1 parameter.
* Add controller for optimizer parameters.
* Add approximation for missing gradient term.
* Get KFAC code?

## Getting Started
If deploying in a remote location (ex. lorraine@cluster12.ais.sandbox):
```
ssh <user@location>
```

Clone the repository:
```
git clone https://github.com/georgeadam/menas.git
```

Create an python 3.6+ environment - for example with conda:
```
conda create --name py36 python=3.6
```

Inside of the python environment install the requirements:
```
pip install -r requirements.txt
```

You may want to run the experiments in tmux:
```
tmux new -s menas
```

Or connect to existing tmux with:
```
tmux a -t menas
```

Experiments can be run from the `menas/src/enas_pytorch` directory:
```
python python train_scripts/train_regular.py --network_type rnn --dataset ptb -entropy_coeff 0.0001 --num_gpu 0
```

Or deployed to the cluster with:
```
srun --gres=gpu:1 -c 2 -l -w dgx1 -p gpuc <COMMAND>
```

Combined:
```
srun --gres=gpu:1 -c 2 -l -w dgx1 -p gpuc python python train_scripts/train_regular.py --network_type rnn --dataset ptb -entropy_coeff 0.0001 --num_gpu 1
```

To make gifs:
```
python generate_gif.py --model_name=<dir> --output=sample.gif
```

To launch tensorboard (maybe within tmux):
```
tensorboard --logdir=logs --port=6006
```

And connect ports on local machine:
```
ssh -N -f -L localhost:16006:localhost:6006 <user@remote>
```

Then go to the following on the local machine:
```
http://localhost:16006
```


## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Cookie Cutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) - Project Template


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

## Authors

* **Jonathan Lorraine** - [lorraine2](https://github.com/lorraine2)
* **Alex Adam** - [georgeadam](https://github.com/georgeadam)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
