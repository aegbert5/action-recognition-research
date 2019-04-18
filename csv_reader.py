import csv
import torch
import itertools as it
import numpy as np
from torch.utils.data import DataLoader
from dataset import RecognitionDataset
import datetime
import cv2
import math
from config import Config
import random

class CSVReader():

    # Retreives all of the features from a csv file into the dataset
    def get_features(self, csv_path, dataset, training_instance):

        # Get the number of features we are looking for
        label_count = dataset.label_count
        feature_count = dataset.feature_count
        num_labels = label_count - feature_count
        
        num_channels = RecognitionDataset.num_channels
        num_joints = RecognitionDataset.num_joints
        num_frames = RecognitionDataset.num_frames
        delimiter = RecognitionDataset.delimiter

        # Fetch the csv data as a numpy array
        csv_data = np.genfromtxt(csv_path, delimiter=delimiter)

        # Convert the 2D data from the csv file to 3D
        #csv_data = np.reshape(csv_data, (-1, 20, 9))
        """For the new dataset, shave off the second skeleton first"""
        csv_data = csv_data[:, np.s_[0:75]]
        csv_data = np.reshape(csv_data, (-1, num_joints, num_channels))

        for i in range(num_labels):

            # Extract information from the current label
            label = dataset.labels[feature_count,0]
            start_frame = dataset.labels[feature_count,1]
            end_frame = dataset.labels[feature_count,2]

            first_index = int(start_frame.item())
            end_index = int(end_frame.item()+1)

            # Get the features for that particular label
            features_raw = csv_data[first_index:end_index]
            time_length, _, _ = features_raw.shape

            # Print error statements for bad raw feature data
            if (time_length <= 0):
                print('ERROR: TIME LENGTH -> {}-{}={} in file #{} {}'.format(start_frame, end_frame, features_raw, training_instance, i))
                break
            if (np.isnan(features_raw).any()):
                print(('ERROR: NAN -> ', training_instance, i))
                break
            if (features_raw.any() == 0):
                print(('ERROR: ZEROS -> ', training_instance, i))
                break

            # Stretch all features to be 100 frames in length
            #features_stretched = cv2.resize(features_raw, (20, 100))
            features_stretched = cv2.resize(features_raw, (num_joints, num_frames))

            # Convert from (100, 20, 9) to (9, 100, 20)
            features_s = np.rollaxis(features_stretched, 2)
            dataset.features[feature_count] = torch.tensor(features_s)
            feature_count += 1

        # Update the number of features found
        dataset.feature_count = feature_count

    # Retreives all of the labels from a csv file into the dataset
    def get_labels(self, csv_path_gt, dataset, training_instance):

        # Get csv file as numpy 2D array (subtract one to reduce range to 0-19)
        csv_data = np.genfromtxt(csv_path_gt, delimiter=',')
        csv_data[:,0] -= 1

        height, width = csv_data.shape
        label_count = dataset.label_count
        
        # Remove the excess column if it exists
        if (width > 3):
            csv_data = csv_data[:, [0,1,2]]

        # Insert the csv data into the desired range
        new_label_count = label_count + height
        dataset.labels[label_count: new_label_count] = torch.tensor(csv_data)

        # Update the label count
        dataset.label_count = new_label_count

    # Stores the features and labels found in two separate files into the dataset
    def get_data(self, csv_path_labels, csv_path_skeleton, dataset, training_instance):

        self.get_labels(csv_path_labels, dataset, training_instance)
        self.get_features(csv_path_skeleton, dataset, training_instance)

    # Returns a dataset given the parameters and configurations
    def get_dataset(self, dataset, instance_ranges, data_directory_path, load_file, features_filepath_save, labels_filepath_save, preprocess_data):

        # Load raw features from .pt files if configured to do so
        if (load_file):
            dataset.features = torch.load(features_filepath_save)
            dataset.labels = torch.load(labels_filepath_save)

            dataset.length, _, _, _ = dataset.features.shape
            dataset.label_count = dataset.length
            dataset.feature_count = dataset.length
        else: # Else scrape them from their individual files
            # Loop over every range of instances, and then loop over every instance in that specific range
            # EX: instance_ranges=[range(1,5), range(6,20), range(100,300), ...]
            for index, range_instance in enumerate(instance_ranges):
                for instance in range_instance:

                    # Creates a 4-character long string of the instance number padded with zeros
                    training_instance = str(instance).zfill(4)

                    label_path = './data/Train_Label_PKU_final/' + training_instance + '-M.txt'
                    skeleton_path = data_directory_path + training_instance + '-M.txt'

                    print((label_path))
                    self.get_data(label_path, skeleton_path, dataset, training_instance)

            torch.save(dataset.features, features_filepath_save)
            torch.save(dataset.labels, labels_filepath_save)
        
        if (preprocess_data):
            dataset.preprocess_features()
                    
        return dataset

    def get_training_loader(self, batch_size, shuffle, load_file, normalize_data):
        
        dataset = RecognitionDataset(7000, True) # There are 7188 instances in total
        ranges = np.array([it.chain(range(2, 247), range(251, 320))])

        data_directory_path = './data/PKU_Skeleton_Renew/'
        features_filepath_save = './features_training.pt'
        labels_filepath_save = './labels_training.pt'

        dataset = self.get_dataset(dataset, ranges, data_directory_path, load_file, features_filepath_save, labels_filepath_save, normalize_data)

        # Wrap dataset in a dataloader
        data_loader = DataLoader(dataset = dataset, batch_size=batch_size, shuffle=shuffle)
        return data_loader
    
    def get_validation_loader(self, batch_size, shuffle, load_file, normalize_data):
        
        dataset = RecognitionDataset(1754, False) # There are 6677 training instances of labels
        ranges = np.array([it.chain(range(320, 365))])

        data_directory_path = './data/PKU_Skeleton_Renew/'
        features_filepath_save = './features_validation.pt'
        labels_filepath_save = './labels_validation.pt'

        dataset = self.get_dataset(dataset, ranges, data_directory_path, load_file, features_filepath_save, labels_filepath_save, normalize_data)

        # Wrap dataset in a dataloader
        data_loader = DataLoader(dataset = dataset, batch_size=batch_size, shuffle=shuffle)
        return data_loader
