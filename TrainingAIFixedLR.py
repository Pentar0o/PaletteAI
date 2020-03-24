import glob
import json
import math

import os
import random
import time
import zipfile
import sys

import cv2

import matplotlib.pyplot as plt
import numpy as np

import gluoncv as gcv
import mxnet as mx

from gluoncv.utils import viz
from mxnet import gluon, nd, autograd
from PIL import Image
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform, SSDDefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric

data_dir = 'DataSet/'
resize_meta = {}

class GTDataset(gluon.data.Dataset):
    """
    Custom Dataset to handle the TrashData Set
    """
    def __init__(self, split='train', data_path=data_dir):
        """
        Parameters
        ---------
        data_path: str, Path to the data folder, default 'data'
        split: str, Which dataset split to request, default 'train'
    
        """
        self.data_path = data_dir
        self.image_info = []
        with open(os.path.join('.', 'output.manifest'), errors='ignore') as f:
            lines = f.readlines()
            for line in lines:
                info = json.loads(line[:-1])
                if len(info['Unknown']['annotations']):
                    self.image_info.append(info)
      
        assert split in ['train', 'test', 'val']
        
        l = len(self.image_info)
        if split == 'train':
            self.image_info = self.image_info[:int(0.8*l)]
        if split == 'val':
            self.image_info = self.image_info[int(0.8*l):int(0.95*l)]
        if split == 'test':
            self.image_info = self.image_info[int(0.95*l):]

    def __getitem__(self, idx):
        """
        Parameters
        ---------
        idx: int, index requested

        Returns
        -------
        image: nd.NDArray
            The image 
        label: np.NDArray bounding box labels of the form [[x1,y1, x2, y2, class], ...]
        """
        info = self.image_info[idx]
        imagename = info['source-ref'].split('/')[-1]
        image = mx.image.imread(os.path.join('ResizedPics', imagename))
        boxes = info['Unknown']['annotations']
        label = []
        for box in boxes:
            label.append([int(box['left']/resize_meta[os.path.join(data_dir, imagename)]),
                          int(box['top']/resize_meta[os.path.join(data_dir, imagename)]),
                          int((box['left']+box['width'])/resize_meta[os.path.join(data_dir, imagename)]),
                          int((box['top']+box['height'])/resize_meta[os.path.join(data_dir, imagename)]),
                          box['class_id']])
        
        return image, np.array(label)
        
    def __len__(self):
        return len(self.image_info)


def resize(picfile, small_edge=512):
    
    """resize image to a small edge of size small_edge
       and save it at same name if smallest edge bigger than small_edge"""
    
    im = Image.open(picfile)
    width, height = im.size
    smallest = min(width, height)

    ratio = smallest / small_edge
    #print('pic ' + picfile + ': applying ratio of ' + str(ratio))
        
    new_width, new_height = int(width/ratio), int(height/ratio)
    #print(new_width, new_height)
    im2 = im.resize((new_width, new_height), Image.ANTIALIAS)
    cheminsauvegarde = 'ResizedPics/' + os.path.basename(picfile)
    im2.save(cheminsauvegarde)
        
    return ratio


def validate(net, val_data, ctx, classes, size):
    """
    Compute the mAP for the network on the validation data
    """
    metric = VOC07MApMetric(iou_thresh=0.5, class_names=classes)
    net.set_nms(0.2)
    for ib, batch in enumerate(val_data):
        
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        det_bboxes, det_ids, det_scores = [],[],[]
        gt_bboxes,gt_ids = [], []
        
        for x, y in zip(data, label):
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            
            metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids[0], None)
    return metric.get()


# Training
def Training(Learning_Rate, NombreEpoch):

    learning_rate = float(Learning_Rate)

    #train_images = glob.glob(data_dir+"*.jpg")
    #print("We have {} images".format(len(train_images)))


    #for pic in train_images:
    #    if pic.lower().endswith('.jpg'):
    #        resize_meta[pic] = resize(pic)

    classes=["sacs","pots","meubles","cartons","matelas","palette","caddy","bidon","chaise","electromenager"]
    train_dataset = GTDataset(split='train')
    validation_dataset = GTDataset(split='val')
    test_dataset = GTDataset(split='test')

    batch_size = 18
    image_size = 512
    num_workers = 8

    num_epochs = NombreEpoch
    
    ctx = [mx.gpu(0)] if mx.context.num_gpus() > 0 else [mx.cpu()]

    net = gcv.model_zoo.get_model('ssd_512_resnet50_v1_coco', pretrained=True)
    net.reset_class(classes)

    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, image_size, image_size)))
    train_transform = SSDDefaultTrainTransform(image_size, image_size, anchors)
    batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
    train_data = gluon.data.DataLoader(train_dataset.transform(train_transform), batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)

    val_transform = SSDDefaultValTransform(image_size, image_size)
    batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_data = gluon.data.DataLoader(validation_dataset.transform(val_transform), batch_size, False, batchify_fn=batchify_fn, last_batch='keep', num_workers=num_workers)

    if (num_epochs > 21):
        steps_epochs = [num_epochs-20, num_epochs-15, num_epochs-10, num_epochs-5]
    else : 
        steps_epochs = [1]

    iterations_per_epoch = math.ceil(len(train_dataset) / batch_size)
    steps_iterations = [s*iterations_per_epoch for s in steps_epochs]
    #print("Learning rate drops after iterations: {}".format(steps_iterations))
    schedule = mx.lr_scheduler.MultiFactorScheduler(step=steps_iterations, factor=0.15)
    schedule.base_lr = learning_rate
    net.collect_params().reset_ctx(ctx)

    trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'learning_rate': schedule.base_lr, 'wd': 0.0004, 'momentum': 0.9, 'lr_scheduler':schedule})

    mbox_loss = gcv.loss.SSDMultiBoxLoss()
    ce_metric = mx.metric.Loss('CrossEntropy')
    smoothl1_metric = mx.metric.Loss('SmoothL1')

    #On cree le tableau pour stocker les mAP et les afficher dans un graphique
    TabHistorymAP = []
    TabEpoch = []
    #On cree les tableaux pour les differentes classes, c'est pas opti, je sais
    TabSac = []
    TabPots = []
    TabMeubles = []
    TabCartons = []
    TabMatelas = []
    TabPalette = []
    TabCaddy = []
    TabBidon = []
    TabChaise = []
    TabElectromenager = []



    best_val = 0 
    for epoch in range(num_epochs):
        net.hybridize(static_alloc=True, static_shape=True)
        #net.cast('float16')
        ce_metric.reset()
        smoothl1_metric.reset()
        tic, btic = time.time(), time.time()

        for i, batch in enumerate(train_data):

            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)

            with autograd.record():
                cls_preds, box_preds = [], []
                for x in data:
                    cls_pred, box_pred, _ = net(x)
                    cls_preds.append(cls_pred)
                    box_preds.append(box_pred)
                sum_loss, cls_loss, box_loss = mbox_loss(cls_preds, box_preds, cls_targets, box_targets)
                autograd.backward(sum_loss)

            trainer.step(1)
            ce_metric.update(0, [l * batch_size for l in cls_loss])
            smoothl1_metric.update(0, [l * batch_size for l in box_loss])
            name1, loss1 = ce_metric.get()
            name2, loss2 = smoothl1_metric.get()

            btic = time.time()
            
        name, val = validate(net, val_data, ctx, classes, image_size)
    
        #A priori on a la moyenne des mAP des classes directement à la fin du tableau
        meanAP = val[10]
        
        #On stock l'hisotrique des résultats par Epoch
        TabHistorymAP.append(meanAP)
        TabEpoch.append(epoch)
        TabSac.append(val[0])
        TabPots.append(val[1])
        TabMeubles.append(val[2])
        TabCartons.append(val[3])
        TabMatelas.append(val[4])
        TabPalette.append(val[5])
        TabCaddy.append(val[6])
        TabBidon.append(val[7])
        TabChaise.append(val[8])
        TabElectromenager.append(val[9])
        
        print('[Epoch {}] Training cost: {:.3f}, Learning rate {}, mAP={:.3f}'.format(epoch, (time.time()-tic), trainer.learning_rate, meanAP))
        
        # If validation accuracy improve, save the parameters
        if meanAP > best_val:
            net.save_parameters('ssd_resnet.trash.params')
            best_val = meanAP
            best_tab = val
            best_epoch = epoch
            print("Saving the parameters, best mAP {}".format(best_val))

    print("Best mAP {}".format(best_val))
    print("Best Epoch : " + str(best_epoch))
    net.save_parameters('ssd_resnet.trash-FixedLR-mAP_'+str(best_val)+'-Epoch_'+str(best_epoch)+'-LR_'+str(learning_rate)+'.params')
    print("Sac : " + str(best_tab[0]))
    print("Pot : " + str(best_tab[1]))
    print("Meuble : " + str(best_tab[2]))
    print("Carton : " + str(best_tab[3]))
    print("Matelas : " + str(best_tab[4]))
    print("Palette : " + str(best_tab[5]))
    print("Caddy : " + str(best_tab[6]))
    print("Bidon : " + str(best_tab[7]))
    print("Chaise : " + str(best_tab[8]))
    print("Electromenager : " + str(best_tab[9]))

def main(Learning_Rate, Epoch, Iterations, AccroissementLR):
    ComputeLR = float(Learning_Rate)
    Nb_Epoch = int(Epoch)
    Iterations = int(Iterations)
    LR_Increase = float(AccroissementLR)

    i = 0

    #On Resize les Images
    train_images = glob.glob(data_dir+"*.jpg")
    print("On a {} images".format(len(train_images)))

    for pic in train_images:
        if pic.lower().endswith('.jpg'):
            resize_meta[pic] = resize(pic)


    #On lance la boucle de training
    while(i<=Iterations):
        print("Batch LR : %f" %ComputeLR)
        Training(ComputeLR,Nb_Epoch)
        ComputeLR += LR_Increase
        i += 1


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('USAGE: {} Learning_Rate Epoch Iterations Accroissement_LR'.format(sys.argv[0]))
    else:
        t1 = time.time()
        main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
        print('Temps de Traitement : %d minutes'%((time.time()-t1)/60))
