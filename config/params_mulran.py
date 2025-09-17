import os


class ModelParams:
    def __init__(self):
        base_root = '/home/octane17/LRFusionPR'
        # Sejong training
        self.info_root = os.path.join(base_root, "data/mulran_infos-sejong.pkl")
        self.bev_dataset_root = os.path.join(base_root, "data/sejong_multi_bev")
        self.database_index_list = [os.path.join(base_root, "data/sejong_db.npy"),
                                    os.path.join(base_root, "data/sejong_train_query.npy")]
        self.train_query_index_list = [os.path.join(base_root, "data/sejong_db.npy"),
                                       os.path.join(base_root, "data/sejong_train_query.npy")]
        self.test_query_index_list = [os.path.join(base_root, "data/sejong_test_query.npy")]

        # # Sejong evaluation
        # self.info_root = os.path.join(base_root, "data/mulran_infos-sejong.pkl")
        # self.bev_dataset_root = os.path.join(base_root, "data/sejong_multi_bev")
        # self.database_index_list = [os.path.join(base_root, "data/sejong_test_db.npy")]
        # self.test_query_index_list = [os.path.join(base_root, "data/sejong_test_query.npy")]

        # # Riverside evaluation
        # self.info_root = os.path.join(base_root, "data/mulran_infos-riverside.pkl")
        # self.bev_dataset_root = os.path.join(base_root, "data/riverside_multi_bev")
        # self.database_index_list = [os.path.join(base_root, "data/riverside_test_db.npy")]
        # self.test_query_index_list = [os.path.join(base_root, "data/riverside_test_query.npy")]

        # # DCC evaluation
        # self.info_root = os.path.join(base_root, "data/mulran_infos-dcc.pkl")
        # self.bev_dataset_root = os.path.join(base_root, "data/dcc_multi_bev")
        # self.database_index_list = [os.path.join(base_root, "data/dcc_test_db.npy")]
        # self.val_query_index_list = [os.path.join(base_root, "data/dcc_test_query.npy")]
        # self.test_query_index_list = [os.path.join(base_root, "data/dcc_test_query.npy")]

        self.checkpoint_path = "/home/octane17/LRFusionPR/weights/scanning_radar.pth.tar"
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
        self.epochs = 10
        self.learning_rate = 0.00005
        self.pos_num = 1
        self.neg_num = 10
        self.pos_dist_threshold = 9
        self.neg_dist_threshold = 18
        self.gth_dist_threshold = 9
        self.loss_margin = 0.5
        self.num_workers = 1

        self.output_dim = 512
