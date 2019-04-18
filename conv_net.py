import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import csv
import os
import itertools as it
import torchnet
import math
import visdom_utils
from config import Config

from csv_reader import CSVReader
from dataset import RecognitionDataset

from sklearn.metrics import confusion_matrix

import net_architectures as architectures

import torchvision

def run_network(network, optimizer, criterion_ce, device, features, labels, batch_count, is_train):

    input_tensor = features.to(device)#cuda()#(async = True)
    labels_tensor = labels[:,0].to(device)#.cuda()#(async = True)

    input_var = torch.autograd.Variable(input_tensor)
    labels_var = torch.autograd.Variable(labels_tensor)

    output_tensor = network(input_var)
    loss_ce = criterion_ce(output_tensor, labels_var)

    # Only backprop when in training mode
    if (is_train):
        optimizer.zero_grad()
        loss_ce.backward()
        optimizer.step()

    # Calculate the accuracy
    output_np = output_tensor.cpu().data.numpy()
    target_np = labels_var.cpu().data.numpy()

    predicted_np = np.argmax(output_np, axis=-1)

    #This can only happen if the feature data is in an invalid format (such as 'nan')
    if (math.isnan(loss_ce)):
        print("ERROR: Loss is undefined")
        print("Set Config.batch_size to 1 to view the data that caused this")
        print('#_batch={}\n#_label={}\n#features={}'.format(batch_count, labels, features))
        return

    return loss_ce, target_np, predicted_np

def get_validation_accuracy(network, optimizer, criterion_ce, device, validation_loader, confusion_graph_validation):

    batch_count = 1

    total = 0
    total_correct = 0

    for data in validation_loader:
        features, labels = data

        network.eval()
        with torch.no_grad():
            loss_ce, predicted_np, target_np = run_network(network, optimizer, criterion_ce, device, features, labels, batch_count, False)

            # The predicted and target are flopped in order to make the data on the correct axis

            #confusion_graph_validation.add(torch.from_numpy(target_np), torch.from_numpy(predicted_np))
            confusion_graph_validation.add(torch.from_numpy(predicted_np), torch.from_numpy(target_np))

            total_correct += (predicted_np == target_np).sum()
            total += features.size(0)

            batch_count += 1

    return total_correct, total

def train_network(network, device):

    csv_reader = CSVReader()

    # Load the training data from the csv files
    training_loader = csv_reader.get_training_loader(Config.batch_size, Config.shuffle, Config.load_data_from_file, Config.preprocess_data)
    validation_loader = csv_reader.get_validation_loader(Config.batch_size, Config.shuffle, Config.load_data_from_file, Config.preprocess_data)

    # Converts data into 3D renderings
    if (Config.use_image_generator):
        from skeleton_visualizer import SkeletonVisualizer

        # You need to give the visualizer 2 skeleton instances to display
        skeleton1 = training_loader.dataset.features[0+12].numpy()
        skeleton2 = training_loader.dataset.features[6261+12].numpy()
        render = SkeletonVisualizer(skeleton1, skeleton2)
        return

    optimizer = optim.SGD(net.parameters(), lr=0.01)
    criterion_ce = nn.CrossEntropyLoss(size_average=True)

    num_classes = RecognitionDataset.num_classifications
    class_names = RecognitionDataset.classification_names

    # This plotter is a class than I have created
    visdom_env_name='Plots'
    if (Config.use_visdom):
        plotter = visdom_utils.VisdomLinePlotter(env_name=visdom_env_name)
        confusion_logger_training = torchnet.logger.VisdomLogger('heatmap', env=visdom_env_name, opts={'title': 'Training Confusion Matrix','columnnames': class_names,'rownames': class_names})
        confusion_logger_validation = torchnet.logger.VisdomLogger('heatmap', env=visdom_env_name, opts={'title': 'Validation Confusion Matrix','columnnames': class_names,'rownames': class_names})
    confusion_graph_training = torchnet.meter.ConfusionMeter(num_classes)
    confusion_graph_validation = torchnet.meter.ConfusionMeter(num_classes)

    batch_count = 1


    total_confusion = np.zeros((num_classes, num_classes))

    for epoch in range(Config.max_num_epochs):
        total = 0
        total_correct = 0
        network.train()

        if (Config.use_visdom):
            confusion_graph_training.reset()
            confusion_graph_validation.reset()

        for data in training_loader:
            features, labels = data
            #print((features.min(), features.max()))

            loss_ce, target_np, predicted_np = run_network(network, optimizer, criterion_ce, device, features, labels, batch_count, True)

            confusion = np.swapaxes(confusion_matrix(target_np, predicted_np, labels=np.arange(0,num_classes)), 0, 1)
            total_confusion = np.add(total_confusion,confusion)

            # The predicted and target are flopped in order to make the data on the correct axis
            #confusion_graph_training.add(torch.from_numpy(target_np), torch.from_numpy(predicted_np))
            if (Config.use_visdom):
                confusion_graph_training.add(torch.from_numpy(predicted_np), torch.from_numpy(target_np))

            total_correct += (predicted_np == target_np).sum()
            total += features.size(0)

            batch_count += 1

        # Calculate the network's accuracy every epoch
        validation_correct, validation_total = get_validation_accuracy(network, optimizer, criterion_ce, device, validation_loader, confusion_graph_validation)

        accuracy_validation = 100 * validation_correct / validation_total
        accuracy_training = 100 * total_correct / total
        print('loss={}   TRAINING: #total={}   #correct={}  VALIDATION: #total={}   #correct={}'.format(loss_ce, total, total_correct, validation_total, validation_correct))
        
        # Plot our results to visdom
        if (Config.use_visdom):
            plotter.plot('loss', 'train', 'Class Loss', epoch, loss_ce.item())
            plotter.plot('accuracy', 'train', 'Class Accuracy', epoch, accuracy_training)
            plotter.plot('accuracy', 'validation', 'Class Accuracy', epoch, accuracy_validation)
            confusion_logger_training.log(confusion_graph_training.value())
            confusion_logger_validation.log(confusion_graph_validation.value())


# The program begins here with the selection of a network architecture
if __name__ == '__main__':
    #net = architectures.AlexNet()
    #net = architectures.CNN_1()
    #net = architectures.ResNet(architectures.BasicBlock, [2,2,2,2]) #RESNET18
    net = architectures.ResNet(architectures.BasicBlock, [3,4,6,3]) #RESNET34 THE BEST ONE SO FAR
    #net = architectures.ResNet(architectures.Bottleneck, [3,4,6,3]) #RESNET50
    #net = architectures.ResNet(architectures.Bottleneck, [3,4,23,3]) #RESNET101
    #net = architectures.ResNet(architectures.Bottleneck, [3,8,36,3]) #RESNET152
    
    #net = architectures.DenseNet(num_init_features=64, growth_rate=32, block_config=(6,12,24,16)) #NET121
    #net = architectures.DenseNet(num_init_features=64, growth_rate=32, block_config=(6,12,32,32)) #NET169
    #net = architectures.DenseNet(num_init_features=64, growth_rate=32, block_config=(6,12,48,32)) #NET201
    #net = architectures.DenseNet(num_init_features=96, growth_rate=48, block_config=(6,12,36,24)) #NET161

    # Use a GPU device, if it is available
    device = torch.device("cpu")
    if (torch.cuda.is_available()):
        print("OK: USING GPU")
        device = torch.device("cuda:1") # Connects to the second device (The Tesla GTX)
    
    else:
        print("WARNING: USING CPU")
    
    # Using 1 GPU is faster than using 2 (parallelization slows the network down)
    net = net.to(device)
    train_network(net, device)
