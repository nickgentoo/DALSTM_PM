# PMdeepNN

This is the code associated to the paper:

**LSTM Networks for Data-Aware Remaining Time Prediction of Business Process Instances**, Nicolò Navarin, Beatrice Vincenzi, Mirko Polato and Alessandro Sperduti. In 2017 IEEE Symposium on Deep Learning @ SSCI, to appear.

The preprint of the paper is available in [arXiv](https://arxiv.org/abs/1711.03822).

## Code Example

Show what the library does as concisely as possible, developers should be able to figure out **how** your project solves their problem by looking at the code example. Make sure the API you are showing off is obvious, and that your code is short and concise.

You can run the code with the following command
python LSTM_sequence_mae.py neurons layers dataset

where neurons is the number of LSTM units per layer, layers is the number of layers and dataset is one among: HELPDESK17, BPI12OEA or BPI12

For example:
python LSTM_sequence_mae.py 10 1 HELPDESK17 

The best (validation) parameters for each dataset are reported in the paper.

NOTE: the training of LSTM may be very slow on CPUs, so we suggest to run this code on GPUs instead.

## Installation
The code requires python and the following libraries:
keras

## Contributors

Nicolò Navarin

## License

If you use this code for research, please cite our paper.
This code is released under the Apache licence 2.0.