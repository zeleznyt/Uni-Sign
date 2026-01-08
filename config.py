mt5_path = "./pretrained_weight/mt5-base"

# label paths
train_label_paths = {
                    "CSL_News": "./data/CSL_News/CSL_News_Labels.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.train",
                    "WLASL": "./data/WLASL/labels-2000.train",
                    "Isharah": "./dataset/Isharah/train_isharah.pkl.gz",
                    "YTASL": "/media/zeleznyt/DATA/data/YTASL_small/YT.annotations.train.json"
                    }

dev_label_paths = {
                    "CSL_News": "./data/CSL_News/CSL_News_Labels.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.dev",
                    "WLASL": "./data/WLASL/labels-2000.dev",
                    "Isharah": "./dataset/Isharah/val_isharah.pkl.gz",
                    "YTASL": "/media/zeleznyt/DATA/data/YTASL_small/YT.annotations.dev.json"
                    }

test_label_paths = {
                    "CSL_News": "./data/CSL_News/CSL_News_Labels.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.test",
                    "WLASL": "./data/WLASL/labels-2000.test",
                    "Isharah": "./dataset/Isharah/test_isharah.pkl.gz",
                    "YTASL": "/media/zeleznyt/DATA/data/YTASL_small/YT.annotations.test.json"
                    }


# video paths
rgb_dirs = {
            "CSL_News": './dataset/CSL_News/rgb_format',
            "CSL_Daily": './dataset/CSL_Daily/sentence-crop',
            "WLASL": "./dataset/WLASL/rgb_format",
            "Isharah": "",
            "YTASL": ""
            }

# pose paths
pose_dirs = {
            "CSL_News": './dataset/CSL_News/pose_format',
            "CSL_Daily": './dataset/CSL_Daily/pose_format',
            "WLASL": "./dataset/WLASL/pose_format",
            "Isharah": "dataset/Isharah/pose_format",
            "YTASL": "/media/zeleznyt/DATA/repo/unisign/Uni-Sign/dataset/YTASL/tmp_raw_keypoints_copy"
            }
