import numpy as np

class Config:

    # Configuration for data loader
    batch_size = 32
    shuffle = True
    load_data_from_file = True # Whether to load features/labels from .pt files or from the data/ folder
    preprocess_data = True
    max_num_epochs = 80
    min_num_epochs = 5

    # Channels do you want to remove from the list of features (x,y,z) -> (0,1,2) respectively
    blacklist_channels = np.array([])

    # Joints do you want to remove from the list of features ([] means to use all of them, default)
    blacklist_joints = np.array([]) # 0-19 exist (0.HipCenter, 1.Spine, 2.ShoulderCenter, 3.Head, 4.ShoulderLeft, 5.ElbowLeft, 6.WristLeft, 7.HandLeft, 8.ShoulderRight, 9.ElbowRight, 10.WristRight, 11.HandRight, 12.HipLeft, 13.KneeLeft, 14.AnkleLeft, 15.FootLeft, 16.HipRight, 17.KneeRight, 18.AnkleRight, and 19.FootRight)

    # Visualization Configurations
    use_visdom = False
    use_image_generator = True
