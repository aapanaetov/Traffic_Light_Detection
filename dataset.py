import os
import sys
import yaml
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import PIL
from PIL import Image
import torchvision
import matplotlib.pyplot as plt

# This function just reads annotaions and taken from official github of Bosch Dataset
def get_all_labels(input_yaml, riib=False, clip=True):
    WIDTH = 1280
    HEIGHT = 720
    
    """ Gets all labels within label file
    Note that RGB images are 1280x720 and RIIB images are 1280x736.
    Args:
        input_yaml->str: Path to yaml file
        riib->bool: If True, change path to labeled pictures
        clip->bool: If True, clips boxes so they do not go out of image bounds
    Returns: Labels for traffic lights
    """
    assert os.path.isfile(input_yaml), "Input yaml {} does not exist".format(input_yaml)
    with open(input_yaml, 'rb') as iy_handle:
        images = yaml.load(iy_handle)

    if not images or not isinstance(images[0], dict) or 'path' not in images[0]:
        raise ValueError('Something seems wrong with this label-file: {}'.format(input_yaml))

    for i in range(len(images)):
        images[i]['path'] = os.path.abspath(os.path.join(os.path.dirname(input_yaml),
                                                         images[i]['path']))

        # There is (at least) one annotation where xmin > xmax
        for j, box in enumerate(images[i]['boxes']):
            if box['x_min'] > box['x_max']:
                images[i]['boxes'][j]['x_min'], images[i]['boxes'][j]['x_max'] = (
                    images[i]['boxes'][j]['x_max'], images[i]['boxes'][j]['x_min'])
            if box['y_min'] > box['y_max']:
                images[i]['boxes'][j]['y_min'], images[i]['boxes'][j]['y_max'] = (
                    images[i]['boxes'][j]['y_max'], images[i]['boxes'][j]['y_min'])

        # There is (at least) one annotation where xmax > 1279
        if clip:
            for j, box in enumerate(images[i]['boxes']):
                images[i]['boxes'][j]['x_min'] = max(min(box['x_min'], WIDTH - 1), 0)
                images[i]['boxes'][j]['x_max'] = max(min(box['x_max'], WIDTH - 1), 0)
                images[i]['boxes'][j]['y_min'] = max(min(box['y_min'], HEIGHT - 1), 0)
                images[i]['boxes'][j]['y_max'] = max(min(box['y_max'], HEIGHT - 1), 0)

        # The raw imager images have additional lines with image information
        # so the annotations need to be shifted. Since they are stored in a different
        # folder, the path also needs modifications.
        if riib:
            images[i]['path'] = images[i]['path'].replace('.png', '.pgm')
            images[i]['path'] = images[i]['path'].replace('rgb/train', 'riib/train')
            images[i]['path'] = images[i]['path'].replace('rgb/test', 'riib/test')
            for box in images[i]['boxes']:
                box['y_max'] = box['y_max'] + 8
                box['y_min'] = box['y_min'] + 8
    return images

class BoschDataset(torch.utils.data.Dataset):
    def __init__(self, transforms=None):
        self.transforms = transforms
        all_samples = get_all_labels('datasets/Bosch/train001/train.yaml')
        clean_samples = []
        for i, sample in enumerate(all_samples):
            if sample['boxes']:
                clean_samples.append(sample)
        self.samples = clean_samples
        
#         classes =  ['Green',
#                     'GreenLeft',
#                     'GreenRight',
#                     'GreenStraight',
#                     'GreenStraightLeft',
#                     'GreenStraightRight',
#                     'Red',
#                     'RedLeft',
#                     'RedRight',
#                     'RedStraight',
#                     'RedStraightLeft',
#                     'Yellow',
#                     'off']
        classes_green = ['Green',
                    'GreenLeft',
                    'GreenRight',
                    'GreenStraight',
                    'GreenStraightLeft',
                    'GreenStraightRight']
        classes_red = ['Red',
                    'RedLeft',
                    'RedRight',
                    'RedStraight',
                    'RedStraightLeft']
        
        class_to_label = dict()
        for cl in classes_green:
            class_to_label[cl] = 1
        for cl in classes_red:
            class_to_label[cl] = 2
        class_to_label['Yellow'] = 3
        class_to_label['off'] = 4
        self.class_to_label = class_to_label

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = plt.imread(sample['path'])

        boxes = []
        labels = []
        is_crowd = []
        for box in sample['boxes']:
            boxes.append([box['x_min'], box['y_min'], box['x_max'], box['y_max']])
            labels.append(self.class_to_label[box['label']])
            is_crowd.append(box['occluded'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(is_crowd, dtype=torch.int64)
        
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.samples)
    
class LisaDataset(torch.utils.data.Dataset):
    def __init__(self, transforms=None):
        self.transforms = transforms
        anno_dict = {}
        train_folders = os.listdir('datasets/LISA/Annotations/Annotations/dayTrain/')
        for folder in train_folders:
            anno_path = 'datasets/LISA/Annotations/Annotations/dayTrain/{}/frameAnnotationsBOX.csv'.format(folder)
            anno = pd.read_csv(anno_path, sep=';')

            for index, row in anno.iterrows():
                prefix = 'datasets/LISA/dayTrain/dayTrain/{}/frames/'.format(folder)
                suffix = row['Filename']
                img_path = prefix + suffix[suffix.rfind('/')+1:]
                label = row['Annotation tag']
                x_min = row['Upper left corner X']
                y_min = row['Upper left corner Y']
                x_max = row['Lower right corner X']
                y_max = row['Lower right corner Y']
                bbox = [x_min, y_min, x_max, y_max]
                try:
                    anno_dict[img_path]['label'].append(label)
                    anno_dict[img_path]['bbox'].append(bbox)
                except:
                    tmp_dict = {}
                    tmp_dict['label'] = [label]
                    tmp_dict['bbox'] = [bbox]
                    anno_dict[img_path] = tmp_dict

        class_to_label = dict()
        class_to_label['go'] = 1
        class_to_label['goLeft'] = 1
        class_to_label['stop'] = 2
        class_to_label['stopLeft'] = 2
        class_to_label['warning'] = 3
        class_to_label['warningLeft'] = 3
        self.class_to_label = class_to_label
        self.anno = anno_dict
        self.frames = sorted(anno_dict.keys())

    def __getitem__(self, idx):
        sample = self.anno[self.frames[idx]]
        img = plt.imread(self.frames[idx]) / 255.

        boxes = sample['bbox']
        labels = [self.class_to_label[cl] for cl in sample['label']]
        is_crowd = [0] * len(labels)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(is_crowd, dtype=torch.int64)
        
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img.float(), target

    def __len__(self):
        return len(self.frames)
    
from engine import train_one_epoch, evaluate
import utils
import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def label_to_class(label):
    l2cl = {}
    l2cl[1] = 'green'
    l2cl[2] = 'red'
    l2cl[3] = 'yellow'
    l2cl[4] = 'unknown'
    return l2cl[label]