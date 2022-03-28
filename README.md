# YAD_STAGIN


### Hyperparameter optimization
* window length
* window stride
* hidden dimension
* sparsity
* lr
* max_lr
* reg_lambda
* num layer
* num_head
* num_epochs
* minibatch size
* readout
* cls_token


### How to use

please refer the utils/option.py file

```
python main.py --dataset=yad_rest --window_siz=25 --window_stride=2 --readout=sero --target=MaDE

```