import os
import csv
import argparse


def parse():
    parser = argparse.ArgumentParser(description='SPATIO-TEMPORAL-ATTENTION-GRAPH-ISOMORPHISM-NETWORK')
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-n', '--exp_name', type=str, default='default')
    parser.add_argument('-k', '--k_fold', type=int, default=5)
    parser.add_argument('-b', '--minibatch_size', type=int, default=3)

    parser.add_argument('-ds', '--sourcedir', type=str, default='./data')
    parser.add_argument('-dt', '--targetdir', type=str, default='./result')
    parser.add_argument('--dataset', type=str, default='yad_rest', choices=['hcp_rest', 'hcp_task', 'yad_rest'])
    
    parser.add_argument('--except_sites', type=str, nargs='+', default=[])
    parser.add_argument('--except_rois', action='store_true')

    parser.add_argument('--target', type=str, default='MaDE', choices=['Gender', 'MaDE', 'PHQ9_total'])
    parser.add_argument('--task', type=str, default='classification', choices=['classification', 'regression'])

    parser.add_argument('--atlas', type=str, default='schaefer400_sub19', choices=['schaefer400_sub19', 'schaefer100_sub19'])
    parser.add_argument('--roi', type=str, default='schaefer', choices=['scahefer', 'aal', 'destrieux', 'harvard_oxford'])
    parser.add_argument('--fwhm', type=float, default=None)

    parser.add_argument('--window_size', type=int, default=25)
    parser.add_argument('--window_stride', type=int, default=2)
    parser.add_argument('--dynamic_length', type=int, default=200)

    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--max_lr', type=float, default=0.001)
    parser.add_argument('--reg_lambda', type=float, default=0.00001)
    parser.add_argument('--clip_grad', type=float, default=0.0)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--sparsity', type=int, default=30)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--readout', type=str, default='sero', choices=['garo', 'sero', 'mean'])
    parser.add_argument('--cls_token', type=str, default='sum', choices=['sum', 'mean', 'param'])

    parser.add_argument('--num_clusters', type=int, default=7)
    parser.add_argument('--subsample', type=int, default=50)

    parser.add_argument('--no_train', action='store_true')
    parser.add_argument('--no_test', action='store_true')
    parser.add_argument('--no_analysis', action='store_true')

    argv = parser.parse_args()
    if argv.exp_name=='default':
        if len(argv.except_sites)==0:
            if argv.except_rois:
                argv.targetdir = os.path.join(argv.targetdir, f"{argv.dataset}_{argv.atlas}_{argv.target}_{argv.readout}_win{ str(argv.window_size) }_stride{ str(argv.window_stride) }_except_rois")
            else:
                argv.targetdir = os.path.join(argv.targetdir, f"{argv.dataset}_{argv.atlas}_{argv.target}_{argv.readout}_win{ str(argv.window_size) }_stride{ str(argv.window_stride) }")
        else:
            argv.targetdir = os.path.join(argv.targetdir, f"{argv.dataset}_{argv.atlas}_{argv.target}_{argv.readout}_win{ str(argv.window_size) }_stride{ str(argv.window_stride) }_excepts{'_'.join(argv.except_sites)}")
    else:
        argv.targetdir = os.path.join(argv.targetdir, argv.exp_name)
    os.makedirs(argv.targetdir, exist_ok=True)
    with open(os.path.join(argv.targetdir, 'argv.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(vars(argv).items())
    return argv
