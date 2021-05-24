from collections import OrderedDict

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from cfg import *
from datasets.bdd import *
from imports import *
from torchvision.utils import save_image
batch_size = 1

from transforms import DomainTransfer
#DT_model = DomainTransfer()
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).cuda()
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, min_size=256, max_size=512, image_mean=[0.5,0.5,0.5], image_std=[0.5,0.5,0.5])
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    ).cuda()  # replace the pre-trained head with a new one
    return model.cuda()


root_img_path = os.path.join(bdd_path, "images", "100k")
root_anno_path = os.path.join(bdd_path, "labels")

val_img_path = root_img_path + "/val/"
val_anno_json_path = root_anno_path + "/bdd100k_labels_images_val.json"

with open("datalists/bdd100k_val_images_path.txt", "rb") as fp:
    bdd_img_path_list = pickle.load(fp)
# bdd_img_path_list = bdd_img_path_list[:10]
val_dataset_bdd = BDD(bdd_img_path_list, val_anno_json_path, transforms=get_transform(train=False))
val_dl_bdd = torch.utils.data.DataLoader(
    val_dataset_bdd,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    collate_fn=utils.collate_fn,
    pin_memory=True,
)

coco_bdd = get_coco_api_from_dataset(val_dl_bdd.dataset)
def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    return iou_types

@torch.no_grad()
def evaluate_(model, coco_dset, data_loader, device):
    iou_types = ["bbox"]
    coco = coco_dset
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cuda")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    model.to(device)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
   
    #to_resize = torchvision.transforms.Resize((256,512))
    to_tensor = torchvision.transforms.ToTensor()
    for image, targets, times in metric_logger.log_every(data_loader, 100, header):

        #image = list(to_tensor(to_resize(img)).to(device) for img in image)
        image = list(img.to(device) for img in image)
       # image = list(DT_model(img.unsqueeze(0).to(device)).squeeze(0) for img in image)
        #save_image(image[0], "./result.jpg")
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        torch.cuda.synchronize()
        model_time = time.time()

        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, outputs)
        }
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator



device = torch.device("cuda")


# num classes 10
model_bdd = get_model(len(val_dataset_bdd.classes))
checkpoint = torch.load("saved_models/" + "/day+noTV_fakenight/bdd100k_15.pth")
model_bdd.load_state_dict(checkpoint["model"])
model_bdd.eval()

for n, p in model_bdd.named_parameters():
    p.requires_grad = False 

for n, p in model_bdd.rpn.named_parameters():
    p.requires_grad = True

for n, p in model_bdd.roi_heads.named_parameters():
    p.requires_grad = True  

print("##########  Evaluation of BDD  ")
evaluate_(model_bdd, coco_bdd, val_dl_bdd, device=torch.device("cuda"))

