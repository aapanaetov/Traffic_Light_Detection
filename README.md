# Traffic_Light_Detection

We present model for traffic light detection based on Faster R-CNN <br />

# Installation
To use this repo you need the following: <br />
-python=3.6 <br /> 
-pytorch=1.8.1 <br />
-torchvision=0.9.1 <br /> 
-cv2=3.4.2 <br />
-matplotlib=3.2.2 <br /> 
-json=2.0.9 <br />
-cython=0.29.21 <br /> 
-yaml=5.4.1 <br /> 
-numpy=1.17.0 <br /> 
-pandas=1.1.3 <br /> 
-pillow=8.0.0 <br /> 
-pycocotool=2.0.2 <br />
Be careful, cocoapi does not work with numpy >= 1.18 so please keep numpy version 1.17 <br /> 
The following example can set up all the required above for anaconda environment: <br />
conda create -n py36 python=3.6 <br />
conda install -c pytorch torchvision <br />
conda install -c conda-forge opencv <br /> 
conda install -c conda-forge matplotlib <br />
conda install -c anaconda pandas <br /> 
conda install -c anaconda pillow <br />
pip install pyyaml <br />
conda install -c anaconda cython <br /> 
pip install -U numpy==1.17.0 -i https://pypi.tuna.tsinghua.edu.cn/simple <br /> 
conda install git <br /> 
pip3 install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI" <br />
# Get started
To test our models you can use ‘notebook’. You can download pretrained weights here https://drive.google.com/drive/folders/1n4LkObQbZU6hoj3hLgsIpqV8ZQPUIA6N?usp=sharing. If you want to use training code then please download Bosch Dataset and Lisa Dataset and change paths in dataset.py. Once you download all the required weights and datasets go to ‘notebook’. Use get_model() function to choose the model you want and set pretrained weights. You can explore different models in model.py file, otherwise you can keep default model defined in notebook. After that you can test it using make_prediction() function. Just feed the path to your video and the model. Set save_video=True if you want to get visualization, otherwise keep it default or False. As a result of make_prediction() function it will save .json with predictions for the video and visualized video if u have chosen this option. You can find default paths to them in make_prediction() function and change if needed. Also you can find examples of visualization here: https://drive.google.com/drive/folders/1n4LkObQbZU6hoj3hLgsIpqV8ZQPUIA6N?usp=sharing

# Experiments

Detector architecture <br />

Detector we use is Faster R-CNN from torchvision.models.detection.faster_rcnn. We have 5 classes for our detector: background, green, red, yellow, unknown. We started with small mobilenet_v2 backbone pretrained on ImageNet and got our first results around 0.358 AP_{0.5} on our validation part (from Bosch Dataset). Then we decided to train detector with other backbones with (or without) different pretrains. For that we were training just for 10 epochs on Bosch Dataset and evaluated them on validation part from Bosch Dataset. To sum up, backbones with COCO pretrain are better than ImageNet pretrain. Also, ImageNet pretrain is better than not having pretrained weights at all as obviously expected. So, we decided to continue with models which have COCO pretrain. Our first larger model with COCO pretrain was with Resnet50FPN backbone. It got significantly higher validation AP than any other model we trained later - 0.723 AP_{0.5} on the same validation part from Bosch Dataset (but we didn't train anything large anymore to be honest because Resnet50FPN had ~10FPS on Nvidia 1080TI and we decided that it is unacceptable to have such a slow inference time and came back to MobileNet family). Then we decided to test inference time of different backbones and decline everything too fast or too slow and keep working on models which have around 30 FPS. Our best candidate was Mobilenet_v3_FPN. It had ~33 FPS, weights pretrained on COCO and good results on validation to start with (after 10 epochs on Bosch Dataset) <br />

Bosch Dataset <br />

For train and validation we use Bosch Dataset (https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset). We removed all images without traffic lights and end up with 3153 images with annotated traffic lights (out of 5093 images in train). Bosch dataset has annotations not only for traffic light color (green/red/yellow), but also left/right/straight/straight left/etc. But we do not use this knowledge and train our model on 5 classes: background, green, red, yellow, unknown. It was a good dataset to start with, but models trained only on Bosch dataset had some issues with distinguishing red and green traffic lights for some reason <br />

Lisa Dataset <br />

Lisa dataset (https://www.kaggle.com/mbornoe/lisa-traffic-light-dataset) is the second dataset we use for training models. It has videos with different scenarios (day/night), but since we do not attempt to solve all the difficult cases and our goal is to have good predictions on casual daytime videos we decided to use only 'easy' party of Lisa dataset - daytime videos. Lisa dataset also has annotations for left/right traffic lights, but we use only color for our model. So to sum up we use 12775 annotated images from Lisa dataset. Overall, Lisa Dataset was a good addition to our train and visually our models become better compare to ones trained only on Bosch Dataset <br />

Other small details <br />

We have tried different augmentations but none of them was good enough: results were almost the same or even worse, so we end up using only flips. We had around 15000 images with multiple traffic lights annotated and we did not train our models from scratch but used already pretrained on COCO weights. So it was enough to finetune our models and augmentations didn’t have such an impressive affect since we didn’t lack data too much. We have used some code from old torchvision versions and cocoapi like evaluation for COCO-style detections, some utils for train, format conversion and others. It was the easier solution since we anyway converted both training datasets to COCO-style for training. Also stackoverfllow and cv2 tutorials helped to set up all the video to frames and frames to video conversion. All the pretrained weights we report about were taken from official sources like torch hub. Others pretrained weights (which were taken from different repositories without license) – we do not report the results of our detector on them but we used them to experiment with different backbones and pretrains (ImageNet/COCO) since not all we wanted to test was on official sources. <br />

Models:<br />

Our best real-time model – Faster R-CNN with Mobilenet_v3_FPN backbone. Pretrained weights were taken from torchvision and you can find them in model.py file. It has inference time about 33 fps and examples of It’s work you can find here. Was trained for 10 epochs on Bosch dataset and then for 4 epochs on Bosch dataset + Lisa Dataset with SGD (lr=5e-3, momentum=0.9, weight decay=5e-4) <br />

Our best model without inference time restrictions is  Faster R-CNN with Resnet50_FPN backbone. The same about backbone – you can find the way to download pretrained weights we were using in model.py file. It has around 10 fps but detection quality is significantly better compare to faster models. Was trained for 10 epochs on Bosch dataset and then for 4 epochs on Bosch dataset + Lisa Dataset with SGD (lr=5e-3, momentum=0.9, weight decay=5e-4). <br />
Weights for these two models you can download right here https://drive.google.com/drive/folders/1n4LkObQbZU6hoj3hLgsIpqV8ZQPUIA6N?usp=sharing <br />
