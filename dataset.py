import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import shuffle, randrange
from torch import tensor, float32, save, load
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import inspect

def prepare_HCPRest_timeseries(atlas='schaefer400_sub19'):
    print_prefix = f"[{inspect.getframeinfo(inspect.currentframe()).function}]"
    timeseries_dir = os.path.join('/u4/surprise/YAD_STAGIN', 'data', 'timeseries')
    sessions = ['REST1', 'REST2']
    phase_encodings = ['RL', 'LR']
    source_dir = f'/u4/HCP/mean_TS/{atlas.split("_")[0]}-yeo17/Atlas_ROIs.2'
    for session in sessions:   ## takes ~1 hr
        for phase in phase_encodings:
            file_list = [file for file in os.listdir(source_dir) if file.endswith(f"{session}_{phase}.419.csv")]
            print(f"{session} {phase} is {len(file_list)}")
            timeseries_dict = {}
            for subject in tqdm(file_list, ncols=60):
                id = subject.split('.')[0]
                # label = hcp_behavior[target_feat][id]
                timeseries = pd.read_csv(os.path.join(source_dir, subject), header=None).to_numpy()
                timeseries_dict[id] = timeseries
            timeseries_file = f'HCP_{session}_{phase}_{atlas}.pth'
            timeseries_path = os.path.join(timeseries_dir, timeseries_file)
            save(timeseries_dict, timeseries_path)
            print(f"{print_prefix} {timeseries_file} is saved.")
    return

def prepare_YADRest_timeseries(atlas='schaefer400_sub19'):
    print_prefix = f'{inspect.getframeinfo(inspect.currentframe()).function}'
    source_dir = {
        'Kaist': f'/u3/Data/YAD_TS/Kaist/rest.FIX_clean_NoiseICs_Censoring_afni/timeseries/{atlas.split("_")[0]}-yeo17/Atlas_ROIs.2/incCbll',
        'SNU': f'/u3/Data/YAD_TS/SNU/rest.FIX_clean_NoiseICs_Censoring/timeseries/{atlas.split("_")[0]}-yeo17/Atlas_ROIs.2/incCbll',
        'Samsung': f'/u3/Data/YAD_TS/Samsung/rest.FIX_clean_NoiseICs_Censoring/timeseries/{atlas.split("_")[0]}-yeo17/Atlas_ROIs.2/incCbll',
        'Gachon': f'/u3/Data/YAD_TS/Gachon/rest.FIX_clean_NoiseICs_Censoring/timeseries/{atlas.split("_")[0]}-yeo17/Atlas_ROIs.2/incCbll',
    }
    timeseries_dir = os.path.join('/u4/surprise/YAD_STAGIN', 'data', 'timeseries')
    timeseries_dict = {}
    for site in source_dir.keys():
        file_list = [file for file in os.listdir(source_dir[site]) if file.endswith(f".csv")]
        print(f"{len(file_list)}")
        for subject in tqdm(file_list, ncols=60):
            id = subject.split('.')[0]
            timeseries_dict[id] = pd.read_csv(os.path.join(source_dir[site], subject), header=None).to_numpy()
    timeseries_file = f'YAD_{atlas}.pth'
    timeseries_path = os.path.join(timeseries_dir, timeseries_file)
    save(timeseries_dict, timeseries_path)
    print(f"{print_prefix} {timeseries_file} is saved.")
    return True

## Dataset class inheriting torch generic "Dataset" class to load HCP resting fMRI data
class DatasetHCPRest(Dataset):
    def __init__(self, atlas='schaefer400_sub19', session='REST1', phase_encoding='RL', target_feature='Gender', k_fold=None):
        print_prefix = f'[{type(self).__name__}.{inspect.getframeinfo(inspect.currentframe()).function}]'
        super().__init__()
        # argparsing
        self.session = session
        self.phase_encoding = phase_encoding
        self.atlas = atlas

        # setting file path
        base_dir = '/u4/surprise/YAD_STAGIN'
        source_dir = f'/u4/HCP/mean_TS/{atlas.split("_")[0]}-yeo17/Atlas_ROIs.2'
        label_path = os.path.join(base_dir, 'data', 'behavior', 'HCP_behavior_data.csv')
        timeseries_dir = os.path.join(base_dir,'data', 'timeseries')
        timeseries_file = f'HCP_{session}_{phase_encoding}_{atlas}.pth'
        timeseries_path = os.path.join(timeseries_dir, timeseries_file)

        if not os.path.exists(timeseries_path):    # no cached file --> caching
            file_list = [file for file in os.listdir(source_dir) if file.endswith(f"{session}_{phase_encoding}.419.csv")]
            print(f"{print_prefix} {session} {phase_encoding} is {len(file_list)}")
            timeseries_dict = {}
            for filename in tqdm(file_list, ncols=60):
                id = filename.split('.')[0]
                timeseries_dict[id] = pd.read_csv(os.path.join(source_dir, filename), header=None).to_numpy()
            save(timeseries_dict, timeseries_path)
            self.timeseries_dict = timeseries_dict
            print(f"{print_prefix} {timeseries_file} is saved.")

        # loading a cached timeseries files
        self.timeseries_dict = load(timeseries_path)
        print(f"{print_prefix} {timeseries_file} is loaded.")

        self.num_nodes, self.num_timepoints = list(self.timeseries_dict.values())[0].shape
        self.full_subject_list = list(self.timeseries_dict.keys())
        if k_fold is None:
            self.subject_list = self.full_subject_list
        else:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0) if k_fold is not None else None
            self.k = None

        # loading the label file
        behavior_df = pd.read_csv(label_path, encoding='CP949').set_index('Subject')
        behavior_df.index = behavior_df.index.astype('str')
        labels_series = behavior_df[target_feature]

        # encoding the labels to integer
        le = LabelEncoder()
        self.labels = le.fit_transform(labels_series).tolist()
        self.labels_dict = { id:label for id,label in zip(labels_series.index.tolist(), self.labels) }
        self.class_names = le.classes_.tolist()
        self.num_classes = len(self.class_names)
        self.full_label_list = [self.labels_dict[subject] for subject in self.full_subject_list]
        print(f"{print_prefix} Done.")

    def __len__(self):
        return len(self.subject_list) if self.k is not None else len(self.full_subject_list)

    def __getitem__(self, idx):
        subject = self.subject_list[idx]
        timeseries = self.timeseries_dict[subject]
        timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / np.std(timeseries, axis=0, keepdims=True)
        label = self.labels_dict[subject]
        return {'id': subject, 'timeseries': tensor(timeseries, dtype=float32), 'label': label}

    def set_fold(self, fold, train=True):
        assert self.k_fold is not None
        self.k = fold
        train_idx, test_idx = list(self.k_fold.split(self.full_subject_list, self.full_label_list))[fold]
        if train: shuffle(train_idx)
        self.subject_list = [self.full_subject_list[idx] for idx in train_idx] if train else [self.full_subject_list[idx] for idx in test_idx]
        return



## Dataset class inheriting torch generic "Dataset" class to load YAD resting fMRI data
class DatasetYADRest(Dataset):
    def __init__(self, atlas='schaefer400_sub19', k_fold=None, target_feature='MaDE'):
        print_prefix = f'[{type(self).__name__}.{inspect.getframeinfo(inspect.currentframe()).function}]'
        super().__init__()
        # argparsing
        self.atlas = atlas

        # setting file path
        base_dir = '/u4/surprise/YAD_STAGIN'
        label_path = os.path.join(base_dir, 'data', 'behavior', 'labelled_modified.csv')
        timeseries_dir = os.path.join(base_dir,'data', 'timeseries')
        timeseries_file = f'YAD_{atlas}.pth'
        timeseries_path = os.path.join(timeseries_dir, timeseries_file)

        if not os.path.exists(timeseries_path):  # no cached file --> caching
            prepare_YADRest_timeseries(atlas=atlas)
            print(f"{print_prefix} {timeseries_file} is saved.")

        self.timeseries_dict = load(timeseries_path)
        self.num_timepoints, self.num_nodes = list(self.timeseries_dict.values())[0].shape
        subjects_with_timeseries = set(self.timeseries_dict.keys())
        print(f"{print_prefix} {timeseries_file} is loaded.")

        # loading the label file
        behavior_df = pd.read_csv(label_path, encoding='CP949').set_index('ID')
        labels_series = behavior_df[target_feature]
        subjects_with_label = set(labels_series.index)
        print(f"{print_prefix} {label_path} is loaded.")

        self.full_subject_list = list( subjects_with_timeseries & subjects_with_label )  # conjoint of subjects_with_timeseries & subjects_with_label
        if k_fold is None:
            self.subject_list = self.full_subject_list
        else:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0) if k_fold is not None else None
            self.k = None

        # encoding the labels to integer
        le = LabelEncoder()
        self.labels = le.fit_transform(labels_series).tolist()
        self.labels_dict = { id:label for id,label in zip(labels_series.index.tolist(), self.labels) }
        self.class_names = le.classes_.tolist()
        print(self.class_names)
        self.num_classes = len(self.class_names)
        print(self.num_classes)
        print(self.full_subject_list)
        self.full_label_list = [self.labels_dict[subject] for subject in self.full_subject_list]
        print(f"{print_prefix} Done.")



    def __len__(self):
        return len(self.subject_list) if self.k is not None else len(self.full_subject_list)


    def __getitem__(self, idx):
        subject = self.subject_list[idx]
        timeseries = self.timeseries_dict[subject]
        timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / np.std(timeseries, axis=0, keepdims=True)
        label = self.labels_dict[subject]
        return {'id': subject, 'timeseries': tensor(timeseries, dtype=float32), 'label': label}


    def set_fold(self, fold, train=True):
        assert self.k_fold is not None
        self.k = fold
        train_idx, test_idx = list(self.k_fold.split(self.full_subject_list, self.full_label_list))[fold]
        if train: shuffle(train_idx)
        self.subject_list = [self.full_subject_list[idx] for idx in train_idx] if train else [self.full_subject_list[idx] for idx in test_idx]

