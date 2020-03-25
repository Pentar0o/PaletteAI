import argparse
import datetime
import glob
import json
import logging
import math
import multiprocessing as mp
import os
import random
import subprocess as sb
import sys
import time
import zipfile
import cv2
import gluoncv as gcv
from gluoncv.utils import viz
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform, SSDDefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from mxnet import gluon, nd, autograd, lr_scheduler
import mxnet as mx
import numpy as np
from PIL import Image


data_dir = 'DataSet/'
resize_meta = {}

class GTDataset(gluon.data.Dataset):
    def __init__(self, split='train', data_path=data_dir):
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
    im = Image.open(picfile)
    width, height = im.size
    smallest = min(width, height)

    ratio = smallest / small_edge
        
    new_width, new_height = int(width/ratio), int(height/ratio)
    im2 = im.resize((new_width, new_height), Image.ANTIALIAS)
    cheminsauvegarde = 'ResizedPics/' + os.path.basename(picfile)
    im2.save(cheminsauvegarde)
        
    return ratio

    
def validate(net, val_data, ctx, classes, size):
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
def Training(Args):
    logging.basicConfig()
            
    args = Args

    # instantiate datasets
    classes=["sacs","pots","meubles","cartons","matelas","palette","caddy","bidon","chaise","electromenager"]
    
    train_dataset = GTDataset(split='train')
    validation_dataset = GTDataset(split='val')
    test_dataset = GTDataset(split='test')
    logging.info("There is {} training images, {} validation images, {} testing images".format(
        len(train_dataset), len(validation_dataset), len(test_dataset)))
    
    
    # instantiate model
    batch_size = args.batch
    image_size = 512
    num_workers = args.workers
    num_epochs = args.epochs
    ctx = [mx.gpu(0)] if mx.context.num_gpus() > 0 else [mx.cpu()]
    
    logging.info('using context ' + str(ctx))
    
    net = gcv.model_zoo.get_model(args.basemodel, pretrained=True)
    net.reset_class(classes)

    
    # instantiate training iterator
    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, image_size, image_size)))
    train_transform = SSDDefaultTrainTransform(image_size, image_size, anchors)
    batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
    train_data = gluon.data.DataLoader(train_dataset.transform(train_transform),
                                       batch_size,
                                       shuffle=True,
                                       batchify_fn=batchify_fn,
                                       last_batch='rollover',
                                       num_workers=num_workers)
    
    
    # instantiate val iterator
    val_transform = SSDDefaultValTransform(image_size, image_size)
    batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_data = gluon.data.DataLoader(validation_dataset.transform(val_transform),
                                     batch_size,
                                     shuffle=False,
                                     batchify_fn=batchify_fn,
                                     last_batch='keep',
                                     num_workers=num_workers)
    
    
    # learning rate scheduler
    scheduler = lr_scheduler.CosineScheduler(
        max_update=int(args.max_update*args.epochs),
        base_lr=args.base_lr,
        final_lr=args.final_lr,
        warmup_steps=int(args.warmup_steps*args.epochs),
        warmup_begin_lr=args.warmup_begin_lr,
        warmup_mode='linear')
    

    # gluon trainer
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(
        net.collect_params(), args.opt,
        {'learning_rate': args.base_lr, 'wd': 0.0004, 'momentum': args.momentum})



    # SSD losses
    mbox_loss = gcv.loss.SSDMultiBoxLoss()
    
    
    # training loop
    best_val = 0 

    for epoch in range(num_epochs):

        trainer.set_learning_rate(scheduler(epoch))
        net.hybridize(static_alloc=True, static_shape=True)
        
        tic = time.time()
    
        for batch in train_data:
    
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
    
            trainer.step(args.batch)
                
        name, val = validate(net, val_data, ctx, classes, image_size)
    
        #le mAP est le dernier élément du tableau
        meanAP = val[10]

        print('[Epoch {}] Training cost: {:.3f}, Learning rate {}, mAP={:.3f}'.format(epoch, (time.time()-tic), trainer.learning_rate, meanAP))

        
        # If validation accuracy improve, save the parameters
        if meanAP > best_val:
            net.save_parameters(args.model_dir + '/ssd_resnet.trash.params')
            best_val = meanAP
            best_epoch = epoch
            best_tab = val
            print('Saving the parameters, best mAP {}'.format(best_val))
    
    net.save_parameters(args.model_dir + '/ssd_resnet.trashai'+'-mAP_'+str(best_val)+'_Lr'+str(args.base_lr)+'-BestEpoch_'+str(best_epoch)+'.params')
    print("Best mAP : {}".format(best_val))
    print("Best Epoch : " + str(best_epoch))
    print("Sac : " + str(best_tab[0]))
    print("Pot : " + str(best_tab[1]))
    print("Meuble : " + str(best_tab[2]))
    print("Carton : " + str(best_tab[3]))
    print("Matelas : " + str(best_tab[4]))
    print("Palette : " + str(best_tab[5]))
    print("Caddy : " + str(best_tab[6]))
    print("Bidon : " + str(best_tab[7]))
    print("Chaise : " + str(best_tab[8]))
    print("Electromeganer : " + str(best_tab[9]))





def main():
    try:
        len(os.environ['SM_MODEL_DIR'])
    except:
        os.environ['SM_MODEL_DIR'] = 'local'
        os.environ['SM_CHANNEL_TRAIN'] = 'local'
    
    #Exemple de ligne de commande pour lancer le script
    #--train ./ --model_dir ./ --save 0 --epochs 20 --workers 8 --batch 15 --basemodel ssd_512_resnet50_v1_coco --base_lr 0.01

    # extract parameters
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--opt', type=str, default='sgd')    
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--final_lr', type=float, default=0.0001)
    parser.add_argument('--warmup_steps', type=float, default=0.2)
    parser.add_argument('--max_update', type=float, default=0.9)
    parser.add_argument('--warmup_begin_lr', type=float, default=0.0001)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--basemodel', type=str, default='ssd_512_mobilenet1.0_voc')
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--increase', type=float, default=0.0001)
    

    args, _ = parser.parse_known_args()


    #On Resize les Images
    train_images = glob.glob(data_dir+"*.jpg")
    print("We have {} images".format(len(train_images)))

    for pic in train_images:
        if pic.lower().endswith('.jpg'):
            resize_meta[pic] = resize(pic)


    #On lance la boucle de training
    while(args.iterations>0):
        print("Batch LR : %f" %args.base_lr)
        Training(args)
        args.base_lr += args.increase
        args.iterations -= 1


if __name__ == '__main__':
    t1 = time.time()
    main()
    print('Temps de Traitement : %d minutes'%((time.time()-t1)/60))
