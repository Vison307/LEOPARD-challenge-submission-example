
import os
import numpy as np
from argparse import Namespace

import torch
from torch.utils.data import DataLoader, Dataset
from inference_utils.tools import read_yaml

import importlib
import inspect
from tqdm import tqdm
import h5py

class InferenceDataset(Dataset):
    def __init__(self, ckpt_path, feature_dir_list):
        super(InferenceDataset, self).__init__()

        self.ckpt_path = ckpt_path
        self.feature_dir_list = feature_dir_list

        self.feature_list = []
        for feature_dir in feature_dir_list:
            self.feature_list.append(sorted(os.listdir(feature_dir)))

    def __len__(self):
        return len(self.feature_list[0])

    def __getitem__(self, idx):
        features = []
        for i in range(len(self.feature_dir_list)):
            feature_dir = self.feature_dir_list[i]
            feature_list = self.feature_list[i]
            filename = feature_list[idx]
            filepath = os.path.join(feature_dir, filename)
            if filepath.endswith('.pt'):
                features.append(torch.load(filepath))
            elif filepath.endswith('.h5'):
                with h5py.File(filepath, 'r') as hdf5_file:
                    # coords = hdf5_file['coords'][()]
                    features.append(torch.tensor(hdf5_file['features'][:]))

        if len(features) == 1:
            path_features = features[0]
        else:
            path_features = torch.cat(features, dim=0)
        # print(f'feature shape: {path_features.shape}')

        return (path_features, filename.split('.tif')[0])

class MultiFeatureInferenceDataset(Dataset):
    def __init__(self, ckpt_path, feature_dir_list):
        super(MultiFeatureInferenceDataset, self).__init__()

        self.ckpt_path = ckpt_path
        self.feature_dir_list = feature_dir_list

        self.feature_list = []
        for feature_dir in feature_dir_list:
            self.feature_list.append(sorted(os.listdir(feature_dir)))

    def __len__(self):
        return len(self.feature_list[0])

    def __getitem__(self, idx):
        features = []
        for i in range(len(self.feature_dir_list)):
            feature_dir = self.feature_dir_list[i]
            feature_list = self.feature_list[i]
            filename = feature_list[idx]
            filepath = os.path.join(feature_dir, filename)
            if filepath.endswith('.pt'):
                features.append(torch.load(filepath))
            elif filepath.endswith('.h5'):
                with h5py.File(filepath, 'r') as hdf5_file:
                    # coords = hdf5_file['coords'][()]
                    features.append(torch.tensor(hdf5_file['features'][:]))

        # print(f'feature shape: {type(features)}')

        return (features, filename.split('.tif')[0])
    
def init_model(args, cfg):
    name = cfg.Model['name']
    try:
        print(name)
        Model = getattr(importlib.import_module(f'models.{name}'), name)
    except:
        raise ValueError('Invalid Module File Name or Invalid Class Name!')
    class_args = inspect.getfullargspec(Model.__init__).args[1:]    # ['i_classifier', 'b_classifier']

    args_dict = {}
    for _args in class_args:
        if _args in cfg.Model.keys():
            args_dict[_args] = cfg.Model[_args]
    model = Model(**args_dict)

    state_dict_path = args.ckpt_path
    state_dict = torch.load(state_dict_path, map_location='cpu')
    model.load_state_dict(state_dict)
    print(f'load checkpoint from {state_dict_path}')
    return model

def collate_MT(batch):

    for item in batch:
        if torch.is_tensor(item[0]):
            img = torch.cat([item[0] for item in batch], dim = 0)
        else:
            img = item[0]
    
    case_id = np.array([item[1] for item in batch])
    return [img, case_id]


def t3(path1, path2, id):
    """
    path1: 2048
    path2: 512
    每个2048的patch, 把余下的512的patch求均值
    """
    # 打开 HDF5 文件
    with h5py.File(path1, 'r') as hdf5_file_2048:
        coords_2048 = hdf5_file_2048['coords'][()]
        features_2048 = hdf5_file_2048['features'][()]

    with h5py.File(path2, 'r') as hdf5_file_512:
        coords_512 = hdf5_file_512['coords'][()]
        features_512 = hdf5_file_512['features'][()]

    # 创建一个数组来存储对齐后的特征
    aligned_features = []
    aligned_coords = []

    # 遍历每个 2048 的 patch
    for i, (x_2048, y_2048) in enumerate(coords_2048):
        feature_2048 = features_2048[i]
        # 找到对应的 512 大小的 patch 的索引
        indices_512 = np.where(
            (coords_512[:, 0] >= x_2048) &
            (coords_512[:, 0] < x_2048 + 2048) &
            (coords_512[:, 1] >= y_2048) &
            (coords_512[:, 1] < y_2048 + 2048)

        )[0]
        
        if len(indices_512) > 0:
            corresponding_features_512 = features_512[indices_512]  # (5, 768)
            # merged_features_512 = np.concatenate(corresponding_features_512, axis=0)
            merged_features_512 = np.mean(corresponding_features_512, axis=0)  # (768,)
            feature_2048 = np.concatenate((feature_2048, merged_features_512), axis=0)   # (1536,)
        else:
            feature_2048 = np.concatenate((feature_2048, feature_2048), axis=0)   # (1536,)

        aligned_features.append(feature_2048)
        aligned_coords.append([x_2048, y_2048])

    aligned_features_array = np.array(aligned_features, dtype=float)
    aligned_coords_array = np.array(aligned_coords)
    # print(aligned_features_array.shape)
    # print(aligned_coords_array.shape)
    # NOTE: to modify hard code
    output_file_name = os.path.join(f'/tmp/featuresC/512with2048/Ctrans/h5_files', id+'.h5')
    if not os.path.exists(os.path.dirname(output_file_name)):
        os.makedirs(os.path.dirname(output_file_name))
    with h5py.File(output_file_name, 'w') as output_file:
        output_file.create_dataset('features', data=aligned_features_array)
        output_file.create_dataset('coords', data=aligned_coords_array)

def main(args, cfg):
    model = init_model(args, cfg)
    model.cuda()
    model.eval()

    if len(args.feature_dir_list) == 1:
        dataset = InferenceDataset(args.ckpt_path, args.feature_dir_list)
    else:
        if 'fusion' not in args.ckpt_path:
            # Concat features along the feature (C) dimension
            h5_dir_512 = args.feature_dir_list[0].replace('pt_files', 'h5_files')
            h5_dir_2048 = args.feature_dir_list[1].replace('pt_files', 'h5_files')
            h5_file_list_512 = sorted(os.listdir(h5_dir_512))
            h5_file_list_2048 = sorted(os.listdir(h5_dir_2048))
            assert len(h5_file_list_512) == len(h5_file_list_2048), f"len(h5_file_list_512): {len(h5_file_list_512)}, len(h5_file_list_2048): {len(h5_file_list_2048)}"
            for i in range(len(h5_file_list_512)):
                t3(os.path.join(h5_dir_2048, h5_file_list_2048[i]), os.path.join(h5_dir_512, h5_file_list_512[i]), h5_file_list_512[i].split('.h5')[0])
            args.feature_dir_list = ['/tmp/featuresC/512with2048/Ctrans/h5_files']
            dataset = InferenceDataset(args.ckpt_path, args.feature_dir_list)
        else:
            # print(f'MultiFeatureInferenceDataset: {args.feature_dir_list}')
            dataset = MultiFeatureInferenceDataset(args.ckpt_path, args.feature_dir_list)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_MT)
    result_list = []
    with torch.no_grad():
        for batch_idx, (path_features, case_id) in enumerate(tqdm(data_loader)):
            if isinstance(path_features, list):
                path_features[0] = path_features[0].cuda()
                path_features[1] = path_features[1].cuda()
            else:
                path_features = path_features.cuda()
            # print(f'type(path_features): {type(path_features)}')
            if cfg.Model.name == 'DSMIL':
                max_prediction, bag_prediction, hazards_i, S_i, hazards_b, S_b = model(wsi=path_features)
                surv_logits = 0.5*torch.sigmoid(max_prediction)+0.5*torch.sigmoid(bag_prediction)
                hazards = torch.sigmoid(surv_logits)
                S = torch.cumprod(1 - surv_logits, dim=1)
            else:
                surv_logits, hazards, S = model(wsi=path_features)
            risk = -torch.sum(S, dim=1).detach().cpu().numpy().tolist()[0]
            predicted_time = -risk
            # predicted_time = torch.argmin(hazards, dim=1).squeeze().item()
            print(f'Predicted time for {case_id} is {predicted_time}; hazards: {hazards}')

            result_list.append({'case_id': case_id[0].split('.pt')[0], 'predicted_time': predicted_time})

    return result_list
                

def inference(config, feature_dir_list, ckpt_path, num_class=4, batch_size=1):
    args = Namespace()
    args.config = config
    args.ckpt_path = ckpt_path
    args.feature_dir_list = feature_dir_list
    args.num_class = num_class
    args.batch_size = batch_size
    
    cfg = read_yaml(args.config)
    return main(args, cfg)

if __name__ == '__main__':
    config = "/data115_2/fzj/LEOPARD/code/LEOPARD/config/DSMIL.yaml"
    ckpt_path = "/data115_2/jsh/LEOPARD/results/DSMIL/0/epoch_21_index_0.7053072625698324.pth"
    feature_dir = "/data115_2/LEOPARD/FEATURES/Leopard_256_at_0.25mpp/pt_files/"
    
    inference(config, feature_dir, ckpt_path)

