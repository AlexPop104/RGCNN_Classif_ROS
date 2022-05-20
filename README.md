# Point Cloud Classification in ROS using Spectral Graph Convolutional Networks


## Abstract
Making predictions using collections of 3D points is necessary for using ToF data. Robotic Operating System (ROS) permits the creation of a pipeline from the live 3D camera to the preprocessing module, then to network module and finally to the vizualization of I/O data. The classification data requires a single object and a large enough ground floor. The preprocessing removes the floor leaving the object to be classified. The network architecture uses the graph convolution module from Te et all 2018, which creates a graph from the point cloud and uses the graph laplacian matrix to create features from the input. A network is trained with live camera data for classification of a single object.

## Overview
The network architecture is RGCNN from Te 2018.

## Prerequisites

We use ROS-melodic
Open3d==0.3.5
Pytorch geometric

## Starting the ROS Data acquisition pipeline

cd ROS_nodes_ws
catkin_make

In separate terminals in the ROS_nodes_ws folder type source devel/setup.bash

The launch file uses first a voxelization module and then a passthrough filter with adjustable parameters. 
The remove_floor module performes euclidean extraction of the largest plane which should be the floor, leaving only the object.
The open3d_sampling_node.py module performs sampling on the point cloud from the topic /no_floor_out

The topic /no_floor_out contains the point cloud without the floor which must be used for classification.
The topic /Segmented_Point_Cloud contains the point cloud sampled at the desired number of points

## ROS Data using live camera

For live camera, use in different terminals the following:

(for the Kinect camera) In first terminal use:
roslaunch openini2_launch openini2.launch 

In second terminal use:
roslaunch pcl_tutorial tutorial_online.launch



## ROS Data using rosbag point cloud recordings.

For rosbag recording of point clouds, start another terminal in the folder containing the .bag file and use the

First terminal:
roscore

Second terminal:
rosbag play -l   file_recording.bag

Third terminal:
roslaunch pcl_tutorial tutorial_online.launch


## Data preparation for training

The classification network needs raw point cloud .pcd files.

The data loader needs the data to be placed in categories and each category MUST HAVE a separate train and test folder.

### Recording the dataset

In order to gather pcd files from rosbag or from live camera, we use pcl_ros pointcloud_to_pcd on the /no_floor_out topic which contains only the object

pcl_ros pointcloud_to_pcd input:=/no_floor_out

An example of created dataset can be found in data/Dataset

### Preprocessing the dataset

The created Dataset has training and test data but it does not have a fixed number of points for each point cloud.

We can use the data loader to create a preprocessed dataset with a fixed number of points.

The data loader is in dataset_loader.py.

the Data loader uses the class PcdDataset which needs the input Dataset folder path "root" and the output dataset folder path "save_path"

If save_path is given, the data loader will process the dataset and export the resulting point clouds with fixed number of poitns.
If save_path is not give, the data loader will use the preprocessed dataset given in root.

By selecting root=Path("-------------Dataset folder path-----------")
And save_path=("--------------Processed folder path---------------")

Example preprocessing dataset:

num_points = 512
root = Path("/home/alex/Alex_documents/RGCNN_git/Classification/Archive_all/Git_folder/data/Dataset/")
save_path = Path("/home/alex/Alex_documents/RGCNN_git/Classification/Archive_all/Git_folder/data/dataset_resampled_v2/")
process_dataset(root=root, save_path=save_path,  num_points=num_points)

Example loading processed dataset:

root =Path("/home/alex/Alex_documents/RGCNN_git/Vizualization_demos/RGCNN_demo_ws/dataset_resampled/")

num_points = 206
train_dataset = PcdDataset(root,valid=False,points=num_points)
test_dataset = PcdDataset(root,folder="test",points=num_points)
loader_train = DenseDataLoader(train_dataset, batch_size=8)
loader_test = DenseDataLoader(test_dataset, batch_size=8)


## Training the network

The network training is done in RGCNN_Classif_cam.py

The program needs the root of the preprocessed dataset and a path to a folder where to save the models.

The following parameters need to be changed accordingly:

num_points= (nr of points of the input point clouds)
parent_directory=("----- Path to folder containing models--------")
root = Path("-------Preprocessed Dataset folder path-----------")

## Testing the network

Loading and testing a model is done on TEST_RGCNN_cam_data.py

As was for training, the network needs num_points and dataset path "root" and the path for the selected network model.

num_points=num_points= (nr of points of the input point clouds)
root = Path("-------Preprocessed Dataset folder path-----------")
path_saved_model

##PFull classification pipeline


## Full Classification model in ROS

### Setting the number of points in the sampling module

In ROS_nodes_ws/pcl_tutorial/src/open3d_sampling_node.py , the desired number of points (num_points) needs to be selected in order for the sampling module to select the correct number of points from the object point cloud.

num_points= (desired number of points)


### Setting classification node

In ROS_node_classification_only_model_cam_v2.py, the label_to_names dictionary must be updated with values and names according to the number,order and names of the object categories.
Additionaly, the model must be loaded 


label_to_names = {0: 'object0',
                  1: 'object1',
                  2: 'object2',
                  3: 'object3',
                  4: 'object4'}

path_saved_model="-- Path to .pt model --"











