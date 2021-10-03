# brgan
Code for "Adversarially Generating Graphs of Bounded Rank" published at DSAA'21.

Paper link: https://wls.ai/DSAA21_BRGAN


## Preperation

Before running the module, make sure you install the required pip modules in `requirements.txt` by running `pip install -r requirements.txt`.

You should also download the datasets from the GraphRNN repo [here](https://github.com/JiaxuanYou/graph-generation/tree/master/dataset).


## Running

To train the model, do:
```bash
cd src
python main.py
```

Parameters can be set in the form `-arg_name=value`, and metrics are automatically logged to [Weights and Biases](https://wandb.ai/).


## Citation
```
@InProceedings{GeneratingGraphsBoundedRank,
author={Shiao, William and Papalexakis, Evangelos},
booktitle={2021 IEEE 8th International Conference on Data Science and Advanced Analytics (DSAA)}, 
title={Adversarially Generating Graphs of Bounded Rank}, 
year={2021},
volume={},
number={},
pages={},
doi={}}
```

## Acknowledgements

This paper uses code from the [GraphRNN repo](https://github.com/JiaxuanYou/graph-generation).
