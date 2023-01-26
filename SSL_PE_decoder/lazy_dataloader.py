import torch
from torch.utils.data import DataLoader
import numpy as np

def preprocess_train(
        source_dim, target_start_dim, target_end_dim, num_workers=0, env_name='jumper'):

    dataset = TrainDataset(
        source_dim=source_dim, target_start_dim=target_start_dim, target_end_dim=target_end_dim,
        env_name=env_name
    )
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=num_workers
    )
    return dataloader


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, source_dim, target_start_dim, target_end_dim, env_name):
        """
        direction: 0 if horizontal else vertical
        """

        self.list_IDs = np.arange(4352) + 1


    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        x1, x2, x3, x4, x5, x6, _ = create_tensor(
            self.source_dim, self.target_start_dim, self.target_end_dim, ID, self.env_name
        )
        return x1, x2, x3, x4, x5, x6