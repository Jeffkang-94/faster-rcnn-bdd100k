# faster-rcnn-bdd100k
Object detection bdd100k with faster-rcnn

## Prerequisites
- Pytorch >= 1.1
- torchvision >= 0.5

directory structure :
```
-| BDD100k
   -| images
     -| 100k
       -| train
       -| test
       -| val
   -| labels
.......
```

### Setting up Config
By default, all paths and hyperparameters are loaded from `cfg.py`. 
`bdd_path` is a data path, which might need to be changed depending on the user profile.
`idx=0` means, bdd100k datasets will be used during the training.

```python
##########  User specific settings ##########################
bdd_path = "/mnt2/datasets/bdd100k" # datapath


batch_size = 64

num_epochs = 25
lr = 0.001
ckpt = False
model_name = "bdd100k_24.pth"
##############################################################

idx = 0
dset_list = ["bdd100k", "Cityscapes"]
ds = dset_list[idx]
```


## Start to train

### get the datalist
if `idx=0`, you can get the bdd dataset list under the `datalist` folder.
```
python get_datalist.py
-| datalists
  -| bdd100k_train_images_path.txt
  -| bdd100k_val_images_path.txt
```

### Specify the data transform
You can modify the transform composition.  
Refer to `bdd.py` under `datasets` folder.

```python
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    transforms.append(T.resize((256,512)))
    transforms.append(T.RandomHorizontalFlip(0.5))
    
    return T.Compose(transforms)
```

### Training
Support for baseline has been added. Domain adaptive features will be added later.

```
python train_baseline.py
```

### Note
Note that, you have to change the min, max value of size if user wants to use `resize transform`.  
Default option provided by torchvision, has `min_size=800, max_size=1333`.
```python
def get_model(num_classes):
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, min_size=256, max_size=512, image_mean=[0.5,0.5,0.5], image_std=[0.5,0.5,0.5])
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )  # replace the pre-trained head with a new one
    return model.cuda()
```


## Evaluation
Evaluation in performed in COCO format. Users need to specify saved `model_name` in `cfg.py`on which evaluation is supposed to occur.

CocoAPI needs to be compiled. first download it from [here](https://github.com/cocodataset/cocoapi)
```
$ cd cocoapi/PythonAPI
$ python setup.py build_ext install
```
Now evaluation can be performed.

```
$ python3 evaluation_script.py
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.148
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.286
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.129
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.031
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.175
**Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.370**
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.117
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.208
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.219
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.276
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.480


## Pre-trained model
Not supported, yet.

## Example
Not supported, yet.