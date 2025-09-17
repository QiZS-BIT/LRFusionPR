import os


class ModelParams:
    def __init__(self):
        base_root = '/home/octane17/LRFusionPR'
        # BS
        self.info_root = os.path.join(base_root, "data/nuscenes_infos-bs.pkl")
        self.bev_dataset_root = os.path.join(base_root, "data/bs_bev")
        self.database_index_list = [os.path.join(base_root, "data/bs_db.npy")]
        self.train_query_index_list = [os.path.join(base_root, "data/bs_train_query.npy")]
        self.val_query_index_list = [os.path.join(base_root, "data/bs_val_query.npy")]
        self.test_query_index_list = [os.path.join(base_root, "data/bs_test_query.npy")]

        # # SON
        # self.info_root = os.path.join(base_root, "data/nuscenes_infos-son.pkl")
        # self.bev_dataset_root = os.path.join(base_root, "data/son_bev")
        # self.database_index_list = [os.path.join(base_root, "data/son_db.npy")]
        # self.test_query_index_list = [os.path.join(base_root, "data/son_test_query.npy")]

        # # SQ
        # self.info_root = os.path.join(base_root, "data/nuscenes_infos-sq.pkl")
        # self.bev_dataset_root = os.path.join(base_root, "data/sq_bev")
        # self.database_index_list = [os.path.join(base_root, "data/sq_db.npy")]
        # self.test_query_index_list = [os.path.join(base_root, "data/sq_test_query.npy")]

        self.checkpoint_path = os.path.join(base_root, "weights/single_chip_radar.pth.tar")
        self.training_root = ""
        if self.training_root:
            if not os.path.exists(self.training_root):
                os.makedirs(self.training_root)
            self.weights_root = os.path.join(self.training_root, "weights")
            if not os.path.exists(self.weights_root):
                os.makedirs(self.weights_root)
            self.logs_root = os.path.join(self.training_root, "logs")
            if not os.path.exists(self.logs_root):
                os.makedirs(self.logs_root)
            self.cache_root = os.path.join(self.training_root, "cache")
            if not os.path.exists(self.cache_root):
                os.makedirs(self.cache_root)
        self.resume_checkpoint = True
        self.resume_epoch = 0
        self.epochs = 12
        self.learning_rate = 0.00005
        self.pos_num = 1
        self.neg_num = 10
        self.pos_dist_threshold = 9
        self.neg_dist_threshold = 18
        self.gth_dist_threshold = 9
        self.loss_margin = 0.5
        self.num_workers = 1

        self.output_dim = 512
