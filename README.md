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
python main.py --dataset=hcp_rest --window_siz=50 --window_stride=3 --readout=sero --target=Gender

```


### Current performance

|                                          Gender  - Scaefer100_sub19 |    accuracy |      recall |   precision |     roc_auc |
|--------------------------------------------------------------------:|------------:|------------:|------------:|------------:|
|                yad_rest_schaefer100_sub19_Gender_garo_win30_stride1 | 0.928±0.027 | 0.890±0.037 | 0.928±0.039 | 0.962±0.020 |
| yad_rest_schaefer100_sub19_Gender_garo_win30_stride1_exceptsSamsung | 0.942±0.018 | 0.905±0.031 | 0.957±0.035 | 0.973±0.029 |
|                yad_rest_schaefer100_sub19_Gender_sero_win30_stride1 | 0.884±0.074 | 0.890±0.058 | 0.846±0.117 | 0.936±0.053 |
| yad_rest_schaefer100_sub19_Gender_sero_win30_stride1_exceptsSamsung | 0.890±0.029 | 0.863±0.090 | 0.880±0.052 | 0.957±0.030 |

|                                            MaDE  - Scaefer100_sub19 |    accuracy |      recall |   precision |     roc_auc |
|--------------------------------------------------------------------:|------------:|------------:|------------:|------------:|
|                  yad_rest_schaefer100_sub19_MaDE_garo_win30_stride1 | 0.920±0.013 | 0.861±0.060 | 0.909±0.052 | 0.960±0.006 |
|   yad_rest_schaefer100_sub19_MaDE_garo_win30_stride1_exceptsSamsung | 0.931±0.039 | 0.831±0.067 | 0.938±0.081 | 0.961±0.032 |
|                  yad_rest_schaefer100_sub19_MaDE_sero_win30_stride1 | 0.912±0.041 | 0.908±0.044 | 0.852±0.064 | 0.962±0.016 |
|   yad_rest_schaefer100_sub19_MaDE_sero_win30_stride1_exceptsSamsung | 0.936±0.022 | 0.905±0.086 | 0.899±0.068 | 0.959±0.027 |

|                                          Gender  - Scaefer400_sub19 |    accuracy |      recall |   precision |     roc_auc |
|--------------------------------------------------------------------:|------------:|------------:|------------:|------------:|
|                yad_rest_schaefer400_sub19_Gender_garo_win25_stride1 | 0.928±0.037 | 0.910±0.037 | 0.920±0.083 | 0.969±0.023 |
|    yad_rest_schaefer400_sub19_Gender_garo_win25_stride1_except_rois | 0.896±0.026 | 0.870±0.024 | 0.877±0.071 | 0.951±0.023 |
|                yad_rest_schaefer400_sub19_Gender_garo_win25_stride2 | 0.924±0.027 | 0.900±0.055 | 0.914±0.060 | 0.977±0.015 |
|                yad_rest_schaefer400_sub19_Gender_garo_win25_stride3 | 0.880±0.044 | 0.770±0.068 | 0.917±0.057 | 0.952±0.027 |
|                yad_rest_schaefer400_sub19_Gender_garo_win50_stride3 | 0.819±0.037 | 0.630±0.081 | 0.892±0.072 | 0.910±0.050 |
|    yad_rest_schaefer400_sub19_Gender_sero_win25_stride1_except_rois | 0.928±0.035 | 0.910±0.058 | 0.920±0.083 | 0.966±0.032 |
|                yad_rest_schaefer400_sub19_Gender_sero_win25_stride2 | 0.872±0.032 | 0.830±0.068 | 0.847±0.032 | 0.952±0.018 |
|                yad_rest_schaefer400_sub19_Gender_sero_win25_stride3 | 0.900±0.036 | 0.780±0.075 | 0.964±0.046 | 0.948±0.026 |
|                yad_rest_schaefer400_sub19_Gender_sero_win50_stride3 | 0.843±0.077 | 0.720±0.175 | 0.877±0.104 | 0.929±0.049 |

|                                           MaDE  -  Scaefer400_sub19 |    accuracy |      recall |   precision |     roc_auc |
|--------------------------------------------------------------------:|------------:|------------:|------------:|------------:|
|                  yad_rest_schaefer400_sub19_MaDE_garo_win25_stride1 | 0.916±0.048 | 0.852±0.095 | 0.921±0.101 | 0.933±0.034 |
|   yad_rest_schaefer400_sub19_MaDE_garo_win25_stride1_exceptsSamsung | 0.919±0.029 | 0.869±0.042 | 0.877±0.083 | 0.944±0.036 |
|                  yad_rest_schaefer400_sub19_MaDE_garo_win25_stride2 | 0.920±0.022 | 0.920±0.060 | 0.864±0.045 | 0.962±0.022 |
|   yad_rest_schaefer400_sub19_MaDE_garo_win25_stride2_exceptsSamsung | 0.954±0.014 | 0.905±0.058 | 0.945±0.045 | 0.980±0.014 |
|                  yad_rest_schaefer400_sub19_MaDE_garo_win25_stride3 | 0.928±0.020 | 0.851±0.044 | 0.941±0.051 | 0.965±0.027 |
|                  yad_rest_schaefer400_sub19_MaDE_sero_win25_stride1 | 0.916±0.041 | 0.850±0.060 | 0.911±0.090 | 0.946±0.036 |
|      yad_rest_schaefer400_sub19_MaDE_sero_win25_stride1_except_rois | 0.928±0.054 | 0.875±0.082 | 0.919±0.084 | 0.966±0.037 |
|   yad_rest_schaefer400_sub19_MaDE_sero_win25_stride1_exceptsSamsung | 0.937±0.028 | 0.851±0.040 | 0.944±0.074 | 0.965±0.036 |
|                  yad_rest_schaefer400_sub19_MaDE_sero_win25_stride2 | 0.879±0.044 | 0.806±0.104 | 0.850±0.081 | 0.947±0.028 |
|   yad_rest_schaefer400_sub19_MaDE_sero_win25_stride2_exceptsSamsung | 0.907±0.038 | 0.791±0.124 | 0.891±0.016 | 0.976±0.017 |
|                  yad_rest_schaefer400_sub19_MaDE_sero_win25_stride3 | 0.900±0.013 | 0.804±0.032 | 0.903±0.059 | 0.960±0.019 |


### To DO
