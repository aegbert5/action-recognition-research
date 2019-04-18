import csv
import torch
import numpy as np
import sys
import zipfile
import itertools as it

class Metadata_Analysis():

    def get_frame_durations(self, zip_path, csv_path):
        max_num_gestures=20 # Total number of gestures that appear in the video

        # Get the list of gestures from the ground truth and frame activation
        gestures_gt = torch.zeros(max_num_gestures)
        gesture_count = 0
        gesture_max_duration = 0
        gesture_min_duration = sys.maxsize
        with zipfile.ZipFile(zip_path) as myzip:
            with myzip.open(csv_path) as csv_file:
                #with open(csv_path, 'r') as csv_file:
                #csv_gt = csv.reader(csv_file)
                #for row in csv_gt:
                for line in csv_file:
                    row = line.decode('UTF-8').split('\n')[0].split(',')
                    frame_start = int(row[1])
                    frame_end = int(row[2])
                    duration = frame_end - frame_start
                    if (duration > gesture_max_duration):
                        gesture_max_duration = duration

                    if (duration < gesture_min_duration):
                        gesture_min_duration = duration

        return gesture_min_duration, gesture_max_duration

    def get_frame_durations_old(self, csv_path):
        max_num_gestures=20 # Total number of gestures that appear in the video

        # Get the list of gestures from the ground truth and frame activation
        gestures_gt = torch.zeros(max_num_gestures)
        gesture_count = 0
        gesture_max_duration = 0
        gesture_min_duration = sys.maxsize
        with open(csv_path, 'r') as csv_file:
            csv_gt = csv.reader(csv_file)
            for row in csv_gt:
                frame_start = int(row[1])
                frame_end = int(row[2])
                duration = frame_end - frame_start
                if (duration > gesture_max_duration):
                    gesture_max_duration = duration

                if (duration < gesture_min_duration):
                    gesture_min_duration = duration

        return gesture_min_duration, gesture_max_duration

    def get_data(self, csv_path_labels, csv_path_skeleton, csv_path_metadata):
        num_frames = self.get_num_frames(csv_path_metadata)

        features = self.get_features(csv_path_skeleton, num_frames)
        labels = self.get_labels(csv_path_labels)

        return features, labels


    def get_pose_length(self):
        label_ending = '_labels.csv'

        total_min = sys.maxsize
        total_max = 0

        # Read from the list of training zip files, that are two levels deep
        first_range = range(1, 100)
        second_range = range(101, 200)
        third_range = range(200, 301)
        fourth_range = range(301, 401)
        fifth_range = range(401, 471)
        ranges = np.array([first_range, second_range, third_range, fourth_range, fifth_range])

        for index, range_instance in enumerate(ranges):
            file_num = index + 1
            outer_zip_path = './data/train/Train' + str(file_num) + '.zip'

            with zipfile.ZipFile(outer_zip_path) as outer_zip_file:
                for instance in range_instance:
                    num_zeros_to_add = 4 - len(str(instance))
                    training_instance = '';
                    for i in range(num_zeros_to_add):
                        training_instance += '0'
                    training_instance += str(instance)
                    #training_instance = '0002'
                    #current_data_path = training_directory + training_instance + label_ending
                    #frame_min, frame_max = analysis.get_frame_durations(current_data_path)

                    zip_path = 'Sample' + training_instance + '.zip'
                    csv_path = 'Sample' + training_instance + label_ending
                    #print((outer_zip_path, csv_path))
                    with outer_zip_file.open(zip_path) as new_zip_path:
                        frame_min, frame_max = analysis.get_frame_durations(new_zip_path, csv_path)

                    print((outer_zip_path, csv_path, frame_min, frame_max))
                    if (frame_min < total_min):
                        total_min = frame_min
                    if (frame_max > total_max):
                        total_max = frame_max

        # Read from the validation zip file
        for instance in it.chain(range(471, 511), range(516, 540), range(541, 551), range(552, 649), range(651, 652), range(653, 690), range(692, 701)):
            num_zeros_to_add = 4 - len(str(instance))
            training_instance = '';
            for i in range(num_zeros_to_add):
                training_instance += '0'
            training_instance += str(instance)
            #training_instance = '0002'
            #current_data_path = training_directory + training_instance + label_ending
            #frame_min, frame_max = analysis.get_frame_durations(current_data_path)

            zip_path = './data/validation/validation_labels.zip'
            csv_path = 'Sample' + training_instance + label_ending
            frame_min, frame_max = analysis.get_frame_durations(zip_path, csv_path)
            print((csv_path, frame_min, frame_max))
            if (frame_min < total_min):
                total_min = frame_min
            if (frame_max > total_max):
                total_max = frame_max

        print((total_min, total_max))

analysis = Metadata_Analysis()
analysis.get_pose_length()
