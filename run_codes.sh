conda activate YAD_STAGIN
cd /u4/surprise/YAD_STAGIN

#pcuda01-1
python main.py --dataset=yad_rest --window_siz=30 --window_stride=1 --readout=garo --target=MaDE --atlas=schaefer100_sub19 --except_sites=Samsung
#pcuda01-2
python main.py --dataset=yad_rest --window_siz=30 --window_stride=1 --readout=garo --target=Gender --atlas=schaefer100_sub19 --except_sites=Samsung
#pcuda02-1
python main.py --dataset=yad_rest --window_siz=30 --window_stride=1 --readout=sero --target=MaDE --atlas=schaefer100_sub19 --except_sites=Samsung
#pcuda02-2
python main.py --dataset=yad_rest --window_siz=30 --window_stride=1 --readout=sero --target=Gender --atlas=schaefer100_sub19 --except_sites=Samsung