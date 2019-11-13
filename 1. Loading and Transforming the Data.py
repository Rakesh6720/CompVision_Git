#!/usr/bin/env python
# coding: utf-8

# In[15]:


# import the required libraries
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2


# In[16]:


# read in the key point data from the csv file
key_pts_frame = pd.read_csv('data/training_frames_keypoints.csv')

n = 0
# find the filename in the column of the csv
image_name = key_pts_frame.iloc[n, 0]
# convert the key points in the columns to numpy array
key_pts = key_pts_frame.iloc[n, 1:].to_numpy()
# reshape the array
key_pts = key_pts.astype('float').reshape(-1, 2)

print('Image name: ', image_name)
print('Landmarks shape: ', key_pts.shape)
print('First 4 key pts: {}'.format(key_pts[:4]))


# In[17]:


print('Number of images: ', key_pts_frame.shape[0])


# In[18]:


def show_keypoints_only(key_pts):
    """Show keypoints"""
    #plt.imshow(image)
    plt.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='.', c='m')


# In[19]:


# a plot showing only keypoints
# why is it upside down?
n = 0

key_pts = key_pts_frame.iloc[n, 1:].to_numpy()
key_pts = key_pts.astype('float').reshape(-1, 2)

plt.figure(figsize=(5, 5))
show_keypoints_only(key_pts)
plt.show()


# In[20]:


def show_keypoints(image, key_pts):
    """Show image with keypoints"""
    plt.imshow(image)
    plt.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='.', c='m')


# In[21]:


# Display a few different types of images by changing the index n

# select an image by index in our data frame
n = 0
image_name = key_pts_frame.iloc[n, 0]
key_pts = key_pts_frame.iloc[n, 1:].to_numpy()
key_pts = key_pts.astype('float').reshape(-1, 2)

plt.figure(figsize=(5, 5))
show_keypoints(mpimg.imread(os.path.join('data/training/', image_name)), key_pts)
plt.show()


# ## Dataset class and Transformations
# 
# To prepare our data for training, we'll be using PyTorch's Dataset class. Much of this this code is a modified version of what can be found in the [PyTorch data loading tutorial](http://pytorch.org/tutorials/beginner/data_loading_tutorial.html).
# 
# #### Dataset class
# 
# ``torch.utils.data.Dataset`` is an abstract class representing a
# dataset. This class will allow us to load batches of image/keypoint data, and uniformly apply transformations to our data, such as rescaling and normalizing images for training a neural network.
# 
# A sample of our dataset will be a dictionary
# ``{'image': image, 'keypoints': key_pts}``. Our dataset will take an
# optional argument ``transform`` so that any required processing can be
# applied on the sample. We will see the usefulness of ``transform`` in the
# next section.

# In[22]:


from torch.utils.data import Dataset, DataLoader

class FacialKeypointsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations
            root_dir (string): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.key_pts_frame = pd.read_csv(csv_file) # returns Dataframe
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.key_pts_frame)
    
    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir, self.key_pts_frame.iloc[idx, 0])
        
        image = mpimg.imread(image_name)
        
        # if image has an alpha color channel, get rid of it
        if(image.shape[2] == 4):
            image = image[:,:,0:3]
            
        key_pts = self.key_pts_frame.iloc[idx, 1:].to_numpy() # why must this be a numpy array?
        key_pts = key_pts.astype('float').reshape(-1, 2) # what does this function do?
        sample = {'image': image, 'keypoints': key_pts}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample


# In[23]:


# Construct the dataset
face_dataset = FacialKeypointsDataset(csv_file='data/training_frames_keypoints.csv', root_dir='data/training/')

# print some stats about the dataset
print('Length of dataset: ', len(face_dataset))


# In[24]:


# display a few of the images from the dataset
num_to_display = 3

for i in range(num_to_display):
    
    # define the size of images
    fig = plt.figure(figsize=(20,10))
    
    # randomly select a sample
    rand_i = np.random.randint(0, len(face_dataset))
    sample = face_dataset[rand_i]
    
    # print the shape of the image and keypoints
    print(i, sample['image'].shape, sample['keypoints'].shape)
    
    ax = plt.subplot(1, num_to_display, i + 1)
    ax.set_title('Sample #{}'.format(i))
    
    show_keypoints(sample['image'], sample['keypoints'])


# ## Transforms
# 
# Now, the images above are not of the same size, and neural networks often expect images that are standardized; a fixed size, with a normalized range for color ranges and coordinates, and (for PyTorch) converted from numpy lists and arrays to Tensors.
# 
# Therefore, we will need to write some pre-processing code.
# Let's create four transforms:
# 
# -  ``Normalize``: to convert a color image to grayscale values with a range of [0,1] and normalize the keypoints to be in a range of about [-1, 1]
# -  ``Rescale``: to rescale an image to a desired size.
# -  ``RandomCrop``: to crop an image randomly.
# -  ``ToTensor``: to convert numpy images to torch images.

# In[25]:


import torch
from torchvision import transforms, utils
# transforms

class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]"""
    
    def __call__(self, sample):
        # get the image, get the keypoints
        image, key_pts = sample['image'], sample['keypoints']
        
        # copy the image, copy the keypoints
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)
        
        # convert image to grayscale
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # scale color range from [0, 255] to [0, 1]
        image_copy = image_copy/255.0
        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50 --> mean squares law?
        key_pts_copy = (key_pts_copy - 100)/50.0
        
        return {'image': image_copy, 'keypoints': key_pts_copy}
    
class Rescale(object):
    """Rescale the image in a sample to a given size.
    
    Args: 
        output_size (tuple or int): Desired output size. 
            If tuple, output is matched to output_size.  
            If int, smaller of image edges is matched to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
            
            new_h, new_w = int(new_h), int(new_w)
            
            img = cv2.resize(image, (new_w, new_h))
            
            # scale the pts, too
            key_pts = key_pts * [new_h / w, new_h / h]
            
            return {'image': img, 'keypoints': key_pts}
        
class RandomCrop(object):
    """Crop randomly the image in a sample
    
    Args:
        output_size (tuple or int): Desires output size.  
        If int, square crop is made
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
            
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        
        image = image[top: top + new_h,
                      left: left + new_w]
        
        key_pts = key_pts - [left, top]
        
        return {'image': image, 'keypoints': key_pts}
        
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add the third color dimension
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
            # swap color axis because
            # numpy image: H x W x C
            # torch iamge: C X H X W
            image = image.transpose((2, 0, 1))
            
            return {'image': torch.from_numpy(image),
                    'keypoints': torch.from_numpy(key_pts)}
    


# ## Test out the transforms
# 
# As you look at each transform, note that, in this case, **order does matter**. For example, you cannot crop a image using a value smaller than the original image (and the orginal images vary in size!), but, if you first rescale the original image, you can then crop it to any size smaller than the rescaled size.

# In[26]:


rescale = Rescale(100)
crop = RandomCrop(50)
composed = transforms.Compose([Rescale(250),
                               RandomCrop(224)])

# apply the transforms to a sample image
test_num = 500
sample = face_dataset[test_num]

fig = plt.figure()
for i, tx in enumerate([rescale, crop, composed]):
    transformed_sample = tx(sample)
    
    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tx).__name__)
    show_keypoints(transformed_sample['image'], transformed_sample['keypoints'])
    
plt.show()


# ## Create the transformed dataset
# 
# Apply the transforms in order to get grayscale images of the same shape. Verify that your transform works by printing out the shape of the resulting data (printing out a few examples should show you a consistent tensor size).

# In[29]:


# define the data tranform
# order matters! i.e. rescaling should come before a smaller crop
data_transform = transforms.Compose([Rescale(250),
                                     RandomCrop(224),
                                     Normalize(),
                                     ToTensor()])

# create the transformed dataset
transformed_dataset = FacialKeypointsDataset(csv_file='data/training_frames_keypoints.csv',
                                             root_dir='data/training/',
                                             transform=data_transform)


# In[30]:


# print some stats about the transformed data
print('Number of images: ', len(transformed_dataset))

# make sure the sample tensors are the expected size
for i in range(5):
    sample = transformed_dataset[i]
    print(i, sample['image'].size(), sample['keypoints'].size())


# In[ ]:




