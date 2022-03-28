conda activate YAD_STAGIN
cd /u4/surprise/YAD_STAGIN

#pcuda01-1
python main.py --dataset=yad_rest --window_siz=25 --window_stride=2 --readout=sero --target=MaDE
#pcuda01-2
python main.py --dataset=yad_rest --window_siz=25 --window_stride=2 --readout=sero --target=Gender  
#pcuda02-1
python main.py --dataset=yad_rest --window_siz=25 --window_stride=2 --readout=garo --target=MaDE 
#pcuda02-2
python main.py --dataset=yad_rest --window_siz=25 --window_stride=2 --readout=garo --target=Gender  