import torch
import itertools as it
import numpy as np
from torch.utils.data import Dataset
from config import Config
import math
import random

# Remove this (temporary use)
import copy

def normalize_torch(x):

    old_range = (x.max(2, keepdim=True)[0] - x.min(2, keepdim=True)[0])
    normed = (x - x.min(2, keepdim=True)[0]) / old_range

    normed_numpy = normed.numpy()
    if (np.isnan(normed_numpy).any()):
        # Sometimes the data does not change over an action, so division by zero results in "nan"
        print('\nWARNING: NAN Exists in feature information after normalization. Converting to 0...')
        nan_locations = np.argwhere(np.isnan(normed_numpy))

        # Get a report as to where nan is showing up
        nan_channels = np.unique(np.delete(nan_locations, (0, 2, 3), axis=1))
        nan_joints = np.unique(np.delete(nan_locations, (0, 1, 2), axis=1))
        nan_instances = np.unique(np.delete(nan_locations, (1, 2, 3), axis=1))
        print('Instances where NAN appears -> ' + str(nan_instances))
        print('Channels where NAN appears -> ' + str(nan_channels))
        print('Joints where NAN appears -> ' + str(nan_joints))
        normed_numpy = np.nan_to_num(normed_numpy)
    
    print('Feature Pre-Processing: Normalize Actions... OK')
    return torch.tensor(normed_numpy)

def get_rotation_matrix(angles):
    cos_value = torch.cos(angles)
    sin_value = torch.sin(angles)
    num_instances, num_frames = angles.shape
    matricies = torch.zeros((num_instances, num_frames, 3, 3))
    #print('BEFORE')
    for instance in range(num_instances):
        for frame in range(num_frames):
            c = cos_value[instance,frame]
            s = sin_value[instance,frame]
            matrix = torch.tensor([[c,0,s],
                              [0,1,0],
                              [-s,0,c]])
            matricies[instance,frame] = matrix
    #print('AFTER')

    return matricies

def transform_invariance(data_torch, labels_torch, mirror_data, rotation_invariance):
    num_instances, num_channels, num_frames, num_joints = data_torch.shape
    if (mirror_data):
        new_data_torch = torch.zeros(data_torch.shape)

    center = copy.deepcopy(data_torch[:,:,:,0])

    for joint in range(num_joints):
        diff = data_torch[:,:,:,joint] - center
        data_torch[:,:,:,joint] = diff
        
        # Mirror augmentation performed at the same time as transofrm invariance to save time and memory
        # NOTE: The joint information must swap, so as to retain real-world joint relationships
        # EX: right-arm action must be applied to the left-arm, and visa-versa
        if (mirror_data):
            for channel in range(num_channels):
                temp = 0
                if (channel == 0):
                    temp = -diff[:,channel,:]
                else:
                    temp = diff[:,channel,:]

                if (joint == 16): #Leg_Right -> Leg_Left
                    new_data_torch[:,channel,:,12] = temp
                elif (joint == 17):
                    new_data_torch[:,channel,:,13] = temp
                elif (joint == 18):
                    new_data_torch[:,channel,:,14] = temp
                elif (joint == 19):
                    new_data_torch[:,channel,:,15] = temp
                elif (joint == 12): #Leg_Left -> Leg_Right
                    new_data_torch[:,channel,:,16] = temp
                elif (joint == 13):
                    new_data_torch[:,channel,:,17] = temp
                elif (joint == 14):
                    new_data_torch[:,channel,:,18] = temp
                elif (joint == 15):
                    new_data_torch[:,channel,:,19] = temp
                elif (joint == 8): #Arm_Right -> Arm_Left
                    new_data_torch[:,channel,:,4] = temp
                elif (joint == 9):
                    new_data_torch[:,channel,:,5] = temp
                elif (joint == 10):
                    new_data_torch[:,channel,:,6] = temp
                elif (joint == 11):
                    new_data_torch[:,channel,:,7] = temp
                elif (joint == 23):
                    new_data_torch[:,channel,:,21] = temp
                elif (joint == 24):
                    new_data_torch[:,channel,:,22] = temp
                elif (joint == 4): #Arm_Left -> Arm_Right
                    new_data_torch[:,channel,:,8] = temp
                elif (joint == 5):
                    new_data_torch[:,channel,:,9] = temp
                elif (joint == 6):
                    new_data_torch[:,channel,:,10] = temp
                elif (joint == 7):
                    new_data_torch[:,channel,:,11] = temp
                elif (joint == 21):
                    new_data_torch[:,channel,:,23] = temp
                elif (joint == 22):
                    new_data_torch[:,channel,:,24] = temp
                else: #Spine to spine
                    new_data_torch[:,channel,:,joint] = temp
        
    if (mirror_data):
        final_data_torch = torch.cat((data_torch, new_data_torch), 0)
        final_labels_torch = torch.cat((labels_torch, labels_torch), 0)
        print("Feature Pre-Processing: Transform Invariance AND Mirror Augmentation... OK")
    else:
        final_data_torch = data_torch
        final_labels_torch = labels_torch
        print("Feature Pre-Processing: Transform Invariance... OK")

    if (rotation_invariance):
        # Mirroring could increase the number of instances
        if (mirror_data):
            num_instances *= 2

        hip_left = copy.deepcopy(final_data_torch[:,:,:,12])
        hip_right = copy.deepcopy(final_data_torch[:,:,:,16])

        # Calculate the angle along the x/z plane to see how much to rotate the skeleton
        hip_diff = hip_right - hip_left
        hip_angle = np.arctan2(hip_diff[:,2,:],hip_diff[:,0,:])

        matricies = get_rotation_matrix(hip_angle)

        for instance in range(num_instances):
            #print('Joint {} of {}'.format(instance, num_instances))
            for frame in range(num_frames):
                final_data_torch[instance,:,frame,:] = torch.matmul(matricies[instance,frame], final_data_torch[instance,:,frame,:])
        print("Feature Pre-Processing: Rotation Invariance... OK")
        
    return final_data_torch, final_labels_torch

"""# KINECT JOINTS BY INDEX #
SPINEBASE = 0
SPINEMID = 1
NECK = 2
HEAD = 3
SHOULDERLEFT = 4
ELBOWLEFT = 5
WRISTLEFT = 6
HANDLEFT = 7
SHOULDERRIGHT = 8
ELBOWRIGHT = 9
WRISTRIGHT = 10
HANDRIGHT = 11
HIPLEFT = 12
KNEELEFT = 13
ANKLELEFT = 14
FOOTLEFT = 15
HIPRIGHT = 16
KNEERIGHT = 17
ANKLERIGHT = 18
FOOTRIGHT = 19
SPINESHOULDER = 20
HANDTIPLEFT  = 21
THUMBLEFT = 22
HANDTIPRIGHT = 23
THUMBRIGHT = 24
"""

# Creates a random offset for the dx, dy, and dz values
def get_48_random_numbers(degree_offset):
    noise_angle_max = math.radians(degree_offset) # Converted to Radians

    random_array = np.array([random.uniform(-noise_angle_max,noise_angle_max) for i in range(48)])

    return random_array

def get_random_angle_offset(joint_dxyz, xz_offset, y_offset, value):

    joint_dxyz = joint_dxyz.numpy()

    # Each of these contain information over all of the frames of the instance
    dx = joint_dxyz[0]
    dy = joint_dxyz[1]
    dz = joint_dxyz[2]

    # joint vector projected onto x/z plane
    b_vector = np.array([dx, np.zeros(len(dx)), dz])

    a_mag = np.sqrt(np.sum(np.power(joint_dxyz,2),axis=0))
    b_mag = np.sqrt(np.power(b_vector,2).sum(0))
    
    # angle between x/z axes
    theta_1 = np.arctan2(dz, dx)

    # angle between y-axis and x/z plane
    # Fix floting point issue by rounding off 1
    theta_2 = np.arccos(np.clip(np.sum(joint_dxyz*b_vector,axis=0)/(a_mag * b_mag),-1,1))

    theta_2[np.argwhere(dy < 0)] *= -1
    
    theta_1_new = theta_1 + xz_offset
    theta_2_new = theta_2 + y_offset

    diff_xz = (np.cos(theta_2_new) - np.cos(theta_2)) * a_mag
    new_dx = np.cos(theta_1_new) * (b_mag + diff_xz)
    new_dy = np.sin(theta_2_new) * a_mag
    new_dz = np.sin(theta_1_new) * (b_mag + diff_xz)

    return torch.tensor([new_dx,new_dy,new_dz], dtype=torch.float32)

def noise_augmentation(data_torch, labels_torch, num_copies_per_action, degree_offset):
    num_instances, num_channels, num_frames, num_joints = data_torch.shape
    new_data_torch = torch.zeros([num_instances*num_copies_per_action, num_channels, num_frames, num_joints])
    new_labels_torch = torch.zeros([num_instances*num_copies_per_action,3], dtype=torch.int64)

    for instance in range(num_instances):
        
        hip_right = data_torch[instance,:,:,16] - data_torch[instance,:,:,0] #HipRight - SpineBase
        knee_right = data_torch[instance,:,:,17] - data_torch[instance,:,:,16] #KneeRight - HipRight
        ankle_right = data_torch[instance,:,:,18] - data_torch[instance,:,:,17] #AnkleRight - KneeRight
        foot_right = data_torch[instance,:,:,19] - data_torch[instance,:,:,18] #FootRight - AnkleRight
        
        hip_left = data_torch[instance,:,:,12] - data_torch[instance,:,:,0] #HipLeft - SpineBase
        knee_left = data_torch[instance,:,:,13] - data_torch[instance,:,:,12] #KneeLeft - HipLeft
        ankle_left = data_torch[instance,:,:,14] - data_torch[instance,:,:,13] #AnkleLeft - KneeLeft
        foot_left = data_torch[instance,:,:,15] - data_torch[instance,:,:,14] #FootLeft - AnkleLeft

        spine_mid = data_torch[instance,:,:,1] - data_torch[instance,:,:,0] #SpineMid - SpineBase
        spine_shoulder = data_torch[instance,:,:,20] - data_torch[instance,:,:,1] #SpineShoulder - SpineMid
        neck = data_torch[instance,:,:,2] - data_torch[instance,:,:,20] #Neck - SpineShoulder
        head = data_torch[instance,:,:,3] - data_torch[instance,:,:,2] #Head - Neck

        shoulder_right = data_torch[instance,:,:,8] - data_torch[instance,:,:,20] #ShoulderRight - SpineShoulder
        elbow_right = data_torch[instance,:,:,9] - data_torch[instance,:,:,8] #ElbowRight - ShoulderRight
        wrist_right = data_torch[instance,:,:,10] - data_torch[instance,:,:,9] #WristRight - ElbowRight
        hand_right = data_torch[instance,:,:,11] - data_torch[instance,:,:,10] #HandRight - WristRight
        hand_tip_right = data_torch[instance,:,:,23] - data_torch[instance,:,:,11] #HandTipRight - HandRight
        thumb_right = data_torch[instance,:,:,24] - data_torch[instance,:,:,11] #ThumbRight - HandRight
        
        shoulder_left = data_torch[instance,:,:,4] - data_torch[instance,:,:,20] #ShoulderRight - SpineShoulder
        elbow_left = data_torch[instance,:,:,5] - data_torch[instance,:,:,4] #ElbowRight - ShoulderRight
        wrist_left = data_torch[instance,:,:,6] - data_torch[instance,:,:,5] #WristRight - ElbowRight
        hand_left = data_torch[instance,:,:,7] - data_torch[instance,:,:,6] #HandRight - WristRight
        hand_tip_left = data_torch[instance,:,:,21] - data_torch[instance,:,:,7] #HandTipRight - HandRight
        thumb_left = data_torch[instance,:,:,22] - data_torch[instance,:,:,7] #ThumbRight - HandRight

        # NOTE: THE TENSORS ABOVE ARENT FOR A SINGLE FRAME, rather the random angle noise is for the entire instance
        
        rand_numbers = np.zeros([num_copies_per_action, 48])
        for num_copy in range(num_copies_per_action):

            rand_numbers[num_copy] = get_48_random_numbers(degree_offset)
            current_numbers = rand_numbers[num_copy]

            # Create some noise involving natural movements
            hip_right = get_random_angle_offset(hip_right, current_numbers[0], current_numbers[1], 1)
            knee_right = get_random_angle_offset(knee_right, current_numbers[2], current_numbers[3], 2)
            ankle_right = get_random_angle_offset(ankle_right, current_numbers[4], current_numbers[5],3)
            foot_right = get_random_angle_offset(foot_right, current_numbers[6], current_numbers[7],4)
            
            hip_left = get_random_angle_offset(hip_left, current_numbers[8], current_numbers[9],5)
            knee_left = get_random_angle_offset(knee_left, current_numbers[10], current_numbers[11],6)
            ankle_left = get_random_angle_offset(ankle_left, current_numbers[12], current_numbers[13],7)
            foot_left = get_random_angle_offset(foot_left, current_numbers[14], current_numbers[15],8)
            
            spine_mid = get_random_angle_offset(spine_mid, current_numbers[16], current_numbers[17],9)
            spine_shoulder = get_random_angle_offset(spine_shoulder, current_numbers[18], current_numbers[19],10)
            neck = get_random_angle_offset(neck, current_numbers[20], current_numbers[21],11)
            head = get_random_angle_offset(head, current_numbers[22], current_numbers[23],12)
            
            shoulder_right = get_random_angle_offset(shoulder_right, current_numbers[24], current_numbers[25],13)
            elbow_right = get_random_angle_offset(elbow_right, current_numbers[26], current_numbers[27],14)
            wrist_right = get_random_angle_offset(wrist_right, current_numbers[28], current_numbers[29],15)
            hand_right = get_random_angle_offset(hand_right, current_numbers[30], current_numbers[31],16)
            hand_tip_right = get_random_angle_offset(hand_tip_right, current_numbers[32], current_numbers[33],17)
            thumb_right = get_random_angle_offset(thumb_right, current_numbers[34], current_numbers[35],18)
            
            shoulder_left = get_random_angle_offset(shoulder_left, current_numbers[36], current_numbers[37],19)
            elbow_left = get_random_angle_offset(elbow_left, current_numbers[38], current_numbers[39],20)
            wrist_left = get_random_angle_offset(wrist_left, current_numbers[40], current_numbers[41],21)
            hand_left = get_random_angle_offset(hand_left, current_numbers[42], current_numbers[43],22)
            hand_tip_left = get_random_angle_offset(hand_tip_left, current_numbers[44], current_numbers[45],23)
            thumb_left = get_random_angle_offset(thumb_left, current_numbers[46], current_numbers[47],24)
           
            # Now Set the new skeleton with the noise in place
            new_data_torch[instance*num_copies_per_action + num_copy,:,:,0] = data_torch[instance,:,:,0]
            new_data_torch[instance*num_copies_per_action + num_copy,:,:,16] = hip_right + new_data_torch[instance*num_copies_per_action + num_copy,:,:,0]
            new_data_torch[instance*num_copies_per_action + num_copy,:,:,17] = knee_right + new_data_torch[instance*num_copies_per_action + num_copy,:,:,16]
            new_data_torch[instance*num_copies_per_action + num_copy,:,:,18] = ankle_right + new_data_torch[instance*num_copies_per_action + num_copy,:,:,17]
            new_data_torch[instance*num_copies_per_action + num_copy,:,:,19] = foot_right + new_data_torch[instance*num_copies_per_action + num_copy,:,:,18]
            
            new_data_torch[instance*num_copies_per_action + num_copy,:,:,12] = hip_left + new_data_torch[instance*num_copies_per_action + num_copy,:,:,0]
            new_data_torch[instance*num_copies_per_action + num_copy,:,:,13] = knee_left + new_data_torch[instance*num_copies_per_action + num_copy,:,:,12]
            new_data_torch[instance*num_copies_per_action + num_copy,:,:,14] = ankle_left + new_data_torch[instance*num_copies_per_action + num_copy,:,:,13]
            new_data_torch[instance*num_copies_per_action + num_copy,:,:,15] = foot_left + new_data_torch[instance*num_copies_per_action + num_copy,:,:,14]
            
            new_data_torch[instance*num_copies_per_action + num_copy,:,:,1] = spine_mid + new_data_torch[instance*num_copies_per_action + num_copy,:,:,0]
            new_data_torch[instance*num_copies_per_action + num_copy,:,:,20] = spine_shoulder + new_data_torch[instance*num_copies_per_action + num_copy,:,:,1]
            new_data_torch[instance*num_copies_per_action + num_copy,:,:,2] = neck + new_data_torch[instance*num_copies_per_action + num_copy,:,:,20]
            new_data_torch[instance*num_copies_per_action + num_copy,:,:,3] = head + new_data_torch[instance*num_copies_per_action + num_copy,:,:,2]
            
            new_data_torch[instance*num_copies_per_action + num_copy,:,:,8] = shoulder_right + new_data_torch[instance*num_copies_per_action + num_copy,:,:,20]
            new_data_torch[instance*num_copies_per_action + num_copy,:,:,9] = elbow_right + new_data_torch[instance*num_copies_per_action + num_copy,:,:,8]
            new_data_torch[instance*num_copies_per_action + num_copy,:,:,10] = wrist_right + new_data_torch[instance*num_copies_per_action + num_copy,:,:,9]
            new_data_torch[instance*num_copies_per_action + num_copy,:,:,11] = hand_right + new_data_torch[instance*num_copies_per_action + num_copy,:,:,10]
            new_data_torch[instance*num_copies_per_action + num_copy,:,:,23] = hand_tip_right + new_data_torch[instance*num_copies_per_action + num_copy,:,:,11]
            new_data_torch[instance*num_copies_per_action + num_copy,:,:,24] = thumb_right + new_data_torch[instance*num_copies_per_action + num_copy,:,:,11]
            
            new_data_torch[instance*num_copies_per_action + num_copy,:,:,4] = shoulder_left + new_data_torch[instance*num_copies_per_action + num_copy,:,:,20]
            new_data_torch[instance*num_copies_per_action + num_copy,:,:,5] = elbow_left + new_data_torch[instance*num_copies_per_action + num_copy,:,:,4]
            new_data_torch[instance*num_copies_per_action + num_copy,:,:,6] = wrist_left + new_data_torch[instance*num_copies_per_action + num_copy,:,:,5]
            new_data_torch[instance*num_copies_per_action + num_copy,:,:,7] = hand_left + new_data_torch[instance*num_copies_per_action + num_copy,:,:,6]
            new_data_torch[instance*num_copies_per_action + num_copy,:,:,21] = hand_tip_left + new_data_torch[instance*num_copies_per_action + num_copy,:,:,7]
            new_data_torch[instance*num_copies_per_action + num_copy,:,:,22] = thumb_left + new_data_torch[instance*num_copies_per_action + num_copy,:,:,7]

            new_labels_torch[instance*num_copies_per_action + num_copy] = labels_torch[instance]
        
    final_data_torch = torch.cat((data_torch, new_data_torch), 0)
    final_labels_torch = torch.cat((labels_torch, new_labels_torch), 0)
    print("Feature Pre-Processing: Joint Noise Augmentation... OK")
    return final_data_torch, final_labels_torch

def differential_feature_extraction(data_torch, labels_torch):
    num_instances, _, _, _ = data_torch.shape

    for instance in range(num_instances):
        
        hip_right = data_torch[instance,:,:,16] - data_torch[instance,:,:,0] #HipRight - SpineBase
        knee_right = data_torch[instance,:,:,17] - data_torch[instance,:,:,16] #KneeRight - HipRight
        ankle_right = data_torch[instance,:,:,18] - data_torch[instance,:,:,17] #AnkleRight - KneeRight
        foot_right = data_torch[instance,:,:,19] - data_torch[instance,:,:,18] #FoorRight - AnkleRight
        
        hip_left = data_torch[instance,:,:,12] - data_torch[instance,:,:,0] #HipLeft - SpineBase
        knee_left = data_torch[instance,:,:,13] - data_torch[instance,:,:,12] #KneeLeft - HipLeft
        ankle_left = data_torch[instance,:,:,14] - data_torch[instance,:,:,13] #AnkleLeft - KneeLeft
        foot_left = data_torch[instance,:,:,15] - data_torch[instance,:,:,14] #FoorLeft - AnkleLeft

        spine_mid = data_torch[instance,:,:,1] - data_torch[instance,:,:,0] #SpineMid - SpineBase
        spine_shoulder = data_torch[instance,:,:,20] - data_torch[instance,:,:,1] #SpineShoulder - SpineMid
        neck = data_torch[instance,:,:,2] - data_torch[instance,:,:,20] #Neck - SpineShoulder
        head = data_torch[instance,:,:,3] - data_torch[instance,:,:,2] #Head - Neck

        shoulder_right = data_torch[instance,:,:,8] - data_torch[instance,:,:,20] #ShoulderRight - SpineShoulder
        elbow_right = data_torch[instance,:,:,9] - data_torch[instance,:,:,8] #ElbowRight - ShoulderRight
        wrist_right = data_torch[instance,:,:,10] - data_torch[instance,:,:,9] #WristRight - ElbowRight
        hand_right = data_torch[instance,:,:,11] - data_torch[instance,:,:,10] #HandRight - WristRight
        hand_tip_right = data_torch[instance,:,:,23] - data_torch[instance,:,:,11] #HandTipRight - HandRight
        thumb_right = data_torch[instance,:,:,24] - data_torch[instance,:,:,11] #ThumbRight - HandRight
        
        shoulder_left = data_torch[instance,:,:,4] - data_torch[instance,:,:,20] #ShoulderRight - SpineShoulder
        elbow_left = data_torch[instance,:,:,5] - data_torch[instance,:,:,4] #ElbowRight - ShoulderRight
        wrist_left = data_torch[instance,:,:,6] - data_torch[instance,:,:,5] #WristRight - ElbowRight
        hand_left = data_torch[instance,:,:,7] - data_torch[instance,:,:,6] #HandRight - WristRight
        hand_tip_left = data_torch[instance,:,:,21] - data_torch[instance,:,:,7] #HandTipRight - HandRight
        thumb_left = data_torch[instance,:,:,22] - data_torch[instance,:,:,7] #ThumbRight - HandRight

        # Now Set the new skeleton as the differentials
        data_torch[instance,:,:,0] = data_torch[instance,:,:,0] - data_torch[instance,:,:,0]
        data_torch[instance,:,:,16] = hip_right
        data_torch[instance,:,:,17] = knee_right
        data_torch[instance,:,:,18] = ankle_right
        data_torch[instance,:,:,19] = foot_right
        
        data_torch[instance,:,:,12] = hip_left
        data_torch[instance,:,:,13] = knee_left
        data_torch[instance,:,:,14] = ankle_left
        data_torch[instance,:,:,15] = foot_left
        
        data_torch[instance,:,:,1] = spine_mid
        data_torch[instance,:,:,20] = spine_shoulder
        data_torch[instance,:,:,2] = neck
        data_torch[instance,:,:,3] = head
        
        data_torch[instance,:,:,8] = shoulder_right
        data_torch[instance,:,:,9] = elbow_right
        data_torch[instance,:,:,10] = wrist_right 
        data_torch[instance,:,:,11] = hand_right
        data_torch[instance,:,:,23] = hand_tip_right
        data_torch[instance,:,:,24] = thumb_right 
        
        data_torch[instance,:,:,4] = shoulder_left
        data_torch[instance,:,:,5] = elbow_left
        data_torch[instance,:,:,6] = wrist_left
        data_torch[instance,:,:,7] = hand_left
        data_torch[instance,:,:,21] = hand_tip_left
        data_torch[instance,:,:,22] = thumb_left

    print("Feature Pre-Processing: Joint Differentialization... OK")
    return data_torch, labels_torch

# TODO: NOT USED
# Makes every human of the same relative size (PRETTY DIFFICULT)
def scale_invariance(data_torch):
    num_instances, num_channels, num_frames, num_joints = data_torch.shape
    for instance in range(num_instances):
        print('INSTANCE ' + str(instance) + ' of ' + str(num_instances))
        for channel in range(num_channels):
            for frame in range(num_frames):
                for joint in range(num_joints):
                    data_torch[instance,channel,frame,joint] -= data_torch[:,channel,frame,20]

    return data_torch

def remove_blacklist(data_torch):
    data_numpy = data_torch.numpy()

    data_numpy = np.delete(data_numpy, Config.blacklist_channels, axis=1)
    data_numpy = np.delete(data_numpy, Config.blacklist_joints, axis=3)

    result = torch.tensor(data_numpy)
    return result


class RecognitionDataset(Dataset):

    # This is for the Chalearn dataset
    """num_channels = 9
    num_joints = 20
    num_frames = 100
    delimiter=','
    num_classifications = 20"""

    # This is for the PKKUMD dataset
    num_channels = 3
    num_joints = 25
    num_frames = 100
    delimiter=' '
    num_classifications = 51
    classification_names = ["bow", "brushing hair", "brushing teeth", "check time (from watch)", "cheer up", "clapping", "cross hands in front (say stop)", "drink water", "drop", "eat meal/snack", "falling", "giving something to other person", "hand waving", "handshaking", "hopping (one foot jumping)", "hugging other person", "jump up", "kicking other person", "kicking something", "make a phone call/answer phone", "pat on back of other person", "pickup", "playing with phone/tablet", "point finger at the other person", "pointing to something with finger", "punching/slapping other person", "pushing other person", "put on a hat/cap", "put something inside pocket", "reading", "rub two hands together", "salute", "sitting down", "standing up", "take off a hat/cap", "take off glasses", "take off jacket", "take out something from pocket", "taking a selfie", "tear up paper", "throw", "touch back (backache)", "touch chest (stomachache/heart pain)", "touch head (headache)", "touch neck (neckache)", "typing on a keyboard", "use a fan (with hand or paper)/feeling warm", "wear jacket", "wear on glasses", "wipe face", "writing"]

    # Give the max number of training instances expected, unused ones will be trimmed off
    def __init__(self, num_training_instances, is_training):
        self.length = num_training_instances
        self.is_training = is_training

        # This is for the Chalearn dataset
        #self.features = torch.zeros(num_training_instances, 9, 100, 20)

        # This is for the PKKUMD dataset
        self.features = torch.zeros(num_training_instances, RecognitionDataset.num_channels, RecognitionDataset.num_frames, RecognitionDataset.num_joints)
        self.labels = torch.zeros(num_training_instances, 3, dtype=torch.int64)

        self.label_count = 0
        self.feature_count = 0

    def preprocess_features(self):

        self.print_report()

        # Assures that we remove any excess data that was not set
        features_numpy = self.features.numpy()

        # Finds rows where data is equivalent to 0, containining incorrect or unset information
        bad_rows = np.where(features_numpy == 0)
        bad_rows = np.unique(bad_rows[0])

        # Masks the bad row instances away, leaving the indexes of the good_rows
        allocated_rows = np.arange(0, self.length)
        good_rows = np.ma.array(allocated_rows, mask=False)
        good_rows.mask[bad_rows] = True
        good_rows = good_rows.compressed()
        
        # Extracts the good_features and good_labels based off of the used rows
        good_features = remove_blacklist(self.features[good_rows])
        good_labels = self.labels[good_rows]

        # One final sanity check to see if there is another form of bad data present
        num_nans = np.sum(np.isnan(features_numpy))

        if (num_nans > 0):
            print("Feature Pre-Processing: Remove Unused Allocated Features... ERROR")
        else:
            print("Feature Pre-Processing: Remove Unused Allocated Features... OK")

        #print('# Instances containing 0s -> ' + str(len(bad_rows)))
        #print('# Instances containing NAN -> ' + str(num_nans))
        #print('# Instances remaining after removal -> ' + str(len(good_rows)))
        #print((good_features.shape, good_labels.shape))

        # Data Pre-Processing methods
        if (self.is_training):
            good_features, good_labels = copy.deepcopy(transform_invariance(good_features, good_labels, True, True))
            good_features, good_labels = copy.deepcopy(noise_augmentation(good_features, good_labels, 1, 2))
            #good_features, good_labels = copy.deepcopy(differential_feature_extraction(good_features, good_labels))
            #good_features = normalize_torch(good_features)
        else:
            #Copy what the training data does, but do not mirror
            good_features, good_labels = copy.deepcopy(transform_invariance(good_features, good_labels, False, True))
            #good_features, good_labels = copy.deepcopy(differential_feature_extraction(good_features, good_labels))
            #good_features = normalize_torch(good_features)

            #good_features, good_labels = copy.deepcopy(noise_augmentation(good_features, good_labels, 1, 10))

        self.features = good_features
        self.labels = good_labels

        # Assures that we set the proper length of the features for training/validation
        self.length, _, _, _ = self.features.shape

        print('Total # Features for Dataset: {}'.format(self.length))

    # Prints a report about the use of the allocated memory for the dataset
    def print_report(self):
        
        # Records how many features/labels were actually found in the datasets
        unused_feature_count = self.length - self.feature_count
        unused_label_count = self.length - self.label_count
        
        if (self.is_training):
            print('-------- Training Dataset Report -----------')
        else:
            print('------- Validation Dataset Report ----------')

        print('Allocated Instances -> ' + str(self.length))
        print('Used Instances -> ' + str(self.feature_count))
        print('Unused Instances -> ' + str(unused_feature_count))

        # The number of feature instances and label instances do not correlate. Data extraction failed
        if (self.feature_count != self.label_count):
            print('ERROR: #Feature Instances != #Labels')
            print('|->' + str(self.feature_count) + ' != ' + str(self.label_count))

    def __getitem__(self, index):

        feature = self.features[index]
        label = self.labels[index]

        return feature, label

    def __len__(self):
        return self.length

    def __add__(self, other):
        return ConcatDataset([self, other])
