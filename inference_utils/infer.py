
import os
import numpy as np
from argparse import Namespace

import torch
from torch.utils.data import DataLoader, Dataset
from inference_utils.tools import read_yaml

import importlib
import inspect
from tqdm import tqdm

class InferenceDataset(Dataset):
    def __init__(self, feature_dir_list):
        super(InferenceDataset, self).__init__()

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
            features.append(torch.load(filepath))

        path_features = torch.cat(features, dim=0)
        print(f'feature shape: {path_features.shape}')

        return (path_features, filename.split('.tif')[0])

    
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

def main(args, cfg):
    model = init_model(args, cfg)
    model.cuda()
    model.eval()

    dataset = InferenceDataset(args.feature_dir_list)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_MT)

    result_list = []
    with torch.no_grad():
        for batch_idx, (path_features, case_id) in enumerate(tqdm(data_loader)):
            path_features = path_features.cuda()
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

