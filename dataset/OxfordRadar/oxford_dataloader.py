from torch.utils.data import DataLoader, default_collate
from config.params_oxford import ModelParams
from dataset.OxfordRadar.oxford_dataset import QueryDataset


def make_dataloader(working_mode, params: ModelParams):
    if working_mode == 'train':
        raise NotImplementedError("Oxford training set is not available")

    elif working_mode == 'test':
        test_dataset = QueryDataset(
            params.info_root,
            params.bev_dataset_root,
            params.database_index_list,
            params.test_query_index_list,
            params.gth_dist_threshold
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=True
        )

        return test_loader, test_dataset


def make_collate_fn():
    def collate_fn(batch):
        return default_collate(batch)
    return collate_fn
