MENAS
==============================

More Efficient Neural Architecture Search.
One Paragraph of project description goes here.

## TODO

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
    
* Write up missing term and background, etc.
* Ablation Studies.


* Fill out the TODO in-depth.
* Fill out the README.
* Construct a skeleton of the final report from the proposal.
* Get ENAS working and write down notes for deployment.
* Make a super simple ENAS, with maybe 1 parameter.
* Add controller for optimizer parameters.
* Add approximation for missing gradient term.
* Get KFAC code?

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
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
