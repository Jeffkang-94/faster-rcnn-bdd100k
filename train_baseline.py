import pickle

from cfg import *
from datasets.bdd import *
from imports import *
#from model.train_model import Cycle_Model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if ds == "bdd100k":
    print(bdd_path)
    root_img_path = os.path.join(bdd_path,  "images", "100k")
    root_anno_path = os.path.join(bdd_path, "labels")

    train_img_path = root_img_path + "/train/"
    #train_fake_img_path = root_img_path + "/fake_night_noTV_noRefine/"
    val_img_path = root_img_path + "/val/"

    train_anno_json_path = root_anno_path + "/bdd100k_labels_images_train.json"
    val_anno_json_path = root_anno_path + "/bdd100k_labels_images_val.json"

    print("Loading files")

    with open("datalists/bdd100k_train_images_path.txt", "rb") as fp:
        train_img_path_list = pickle.load(fp)
    #with open("datalists/bdd100k_train_fake_images_path.txt", "rb") as fp:
    #    train_fake_img_path_list = pickle.load(fp)
    with open("datalists/bdd100k_val_images_path.txt", "rb") as fp:
        val_img_path_list = pickle.load(fp)

    dataset_train = dset = BDD(
        train_img_path_list, train_anno_json_path, get_transform(train=True)
    )
    #dataset_fake_train = BDD(train_fake_img_path_list, train_anno_json_path, get_transform(train=True))
    dl = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )
    
    dataset_val = BDD(
        val_img_path_list, val_anno_json_path,  get_transform(train=False)
    )
    val_dl = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

print("Loading done")


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    #,min_size=256, max_size=512)
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, min_size=256, max_size=512, image_mean=[0.5,0.5,0.5], image_std=[0.5,0.5,0.5])
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )  # replace the pre-trained head with a new one
    return model.cuda()


print("Model initialization")
model = get_model(len(dataset_train.classes))
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr, max_lr=6e-3)

try:
    os.mkdir("saved_models/")
except:
    pass

start_epoch =0 
if ckpt:
    checkpoint = torch.load("saved_models/bdd100k_5.pth")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"]
# epoch = checkpoint['epoch']


#print("Training started with CycleGaN")



for epoch in tqdm(range(start_epoch, num_epochs)):
    train_one_epoch(model, optimizer, dl,  device, epoch, print_freq=200)
    lr_scheduler.step()
    

    if epoch == 5 or epoch == 10 or epoch == 15 or epoch == 20 or epoch == 24 or epoch==30 or epoch==35 or epoch==40 or epoch==49:
        save_name = "saved_models/bdd100k_" + str(epoch) + ".pth"
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch":epoch},
            save_name,
        )
        print("Saved model", save_name)
        #evaluate(model, val_dl, device)

##### Daytime training#######