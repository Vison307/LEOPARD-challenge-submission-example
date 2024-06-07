import torch
import numpy as np

def collate_MT(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    label_surv = torch.LongTensor([item[1] for item in batch])
    event_time = np.array([item[2] for item in batch])
    c = torch.FloatTensor([item[3] for item in batch])
    case_id = np.array([item[4] for item in batch])
    return [img, label_surv, event_time, c, case_id]