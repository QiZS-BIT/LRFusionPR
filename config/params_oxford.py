import os


class ModelParams:
    def __init__(self):
        self.info_root = "/home/octane17/LRFusionPR/data/oxford_infos-2019-01-11-13-24-51.pkl"
        self.bev_dataset_root = "/home/octane17/LRFusionPR/data/oxford_infos-2019-01-11-13-24-51_bev"
        self.database_index_list = ["/home/octane17/LRFusionPR/data/oxford_infos-2019-01-11-13-24-51_db.npy"]
        self.test_query_index_list = ["/home/octane17/LRFusionPR/data/oxford_infos-2019-01-11-13-24-51_test_query.npy"]

        self.checkpoint_path = "/home/octane17/LRFusionPR/weights/scanning_radar.pth.tar"
        self.resume_checkpoint = True
        self.gth_dist_threshold = 9
        self.num_workers = 1

        self.output_dim = 512
