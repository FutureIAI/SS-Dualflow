import argparse
import os
import time

import torch
import yaml
from ignite.contrib import metrics

import constants as const

import dataset_dream

import Dualflow
import utils

import numpy as np
from sklearn.metrics import roc_auc_score
import cv2

def build_train_data_loader(args, classname):
    train_dataset = dataset_dream.MVTecDRAEMTrainDataset(
        root_dir=args.data +'/'+ classname + "/train/good/",
        anomaly_source_path = '/home/cgz/MZL/dateset/DRAEM/dtd/images/',
        resize_shape=[256, 256]
    )
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

def build_test_data_loader(args, classname):
    test_dataset = dataset_dream.MVTecDRAEMTestDataset(
        root_dir=args.data +'/' + classname + "/test/",
        resize_shape = [256, 256]
    )
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )


def build_model(config):
    model = Dualflow.SS_Dualflow(# 创建模型
        backbone_name=config["backbone_name"],
        flow_steps=config["flow_step"],
        input_size=config["input_size"],
        conv3x3_only=config["conv3x3_only"],
        hidden_ratio=config["hidden_ratio"],
    )
    print(
        "Model A.D. Param#: {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    return model


def build_optimizer(model):
    return torch.optim.Adam(
        model.parameters(), lr=const.LR, weight_decay=const.WEIGHT_DECAY
    )


def train_one_epoch(args,dataloader, model, optimizer, epoch):
    model.train()
    loss_meter = utils.AverageMeter()
    for step, data in enumerate(dataloader):
        image = data["image"].cuda()
        augmented_image = data["augmented_image"].cuda()
        anomaly_mask = data["anomaly_mask"].cuda()
        # forward

        # import time# 计算运行时间
        # torch.cuda.synchronize()
        # start = time.time()
        ret = model(augmented_image,anomaly_mask) #
        # torch.cuda.synchronize()
        # end = time.time()
        # print('infer_time:',end-start)

        loss = ret["loss"]
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log
        loss_meter.update(loss.item())
        if (step + 1) % const.LOG_INTERVAL == 0 or (step + 1) == len(dataloader):
            print(
                "{} Epoch {} - Step {}: loss = {:.3f}({:.3f})".format(
                    args.category,epoch + 1, step + 1, loss_meter.val, loss_meter.avg
                )
            )


def eval_once(args,dataloader, model,epoch,f):
    model.eval()
    auroc_metric = metrics.ROC_AUC()
    for data in dataloader:
        image = data["image"].cuda()
        mask = data["mask"].cuda()
        with torch.no_grad():
            ret = model(image,mask)
        outputs = ret["anomaly_map"].cpu().detach()
        outputs = outputs.flatten()
        mask = mask.flatten()
        auroc_metric.update((outputs, mask))
    auroc = auroc_metric.compute()
    print("{}   AUROC: {}".format(args.category, auroc))
    f.write("{}   epoch: {}".format(args.category,epoch))
    f.write('\n')
    f.write('pixel ROCAUC: %.4f' % (auroc))
    f.write('\n')

def eval_once_2(args,dataloader, model,epoch,f):
    gt_list = []
    score_map = []
    gt_mask_list = []

    model.eval()
    auroc_metric = metrics.ROC_AUC()
    i=0
    for data in dataloader:
        image = data["image"].cuda()
        mask = data["mask"].cuda()
        if mask.size()[1] == 3:
            mask = mask[:, 0, :, :].unsqueeze(0)
        for i in range(mask.size()[0]):
            gt_mask_list.append(mask[i].detach().cpu().numpy())
            k = mask[i].max()
            if k == 1:
                gt_list.append(1)
            else:
                gt_list.append(0)

        with torch.no_grad():
            ret = model(image,mask)
        outputs = ret["anomaly_map"].cpu().detach()

        for i in range(outputs.size()[0]):
            score_map.append(outputs[i].detach().cpu().numpy())

        outputs = outputs.flatten()
        mask = mask.flatten()
        auroc_metric.update((outputs, mask))
        i=i+1
    auroc = auroc_metric.compute()
    print("{}-------AUROC: {}".format(args.category,auroc))

    # Normalization 归一化
    max_score = np.array(score_map).max()
    min_score = np.array(score_map).min()
    scores = (score_map - min_score) / (max_score - min_score)
    # AUC
    # calculate per-pixel level ROCAUC     ---------------------------------
    gt_mask = np.asarray(gt_mask_list)
    scores = np.squeeze(scores)
    per_pixel_rocauc = roc_auc_score(gt_mask.flatten().astype('int'), scores.flatten())
    # calculate image-level ROC AUC score   ---------------------------------
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    gt_list = np.asarray(gt_list)
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    # 输出结果
    print("class_name: {}     epoch: {}".format(args.category, epoch))
    print('pixel ROCAUC: %.4f     image ROCAUC: %.4f' % (
        per_pixel_rocauc, img_roc_auc))

    f.write("{}   epoch: {}".format(args.category, epoch))
    f.write('\n')
    f.write('pixel ROCAUC: %.4f     image ROCAUC: %.4f ' % (
        per_pixel_rocauc, img_roc_auc))
    f.write('\n')
    return [per_pixel_rocauc, img_roc_auc, epoch]


def train(args):
    os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)# 创建一个检测点文件夹
    for classname in const.MVTEC_CATEGORIES:
        args.category = classname
        args.cat = classname


        checkpoint_dir = os.path.join(
            const.CHECKPOINT_DIR, "%s" % classname        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        f = open(os.path.join(checkpoint_dir+ '/' + args.category+'.txt'), 'w')

        config = yaml.safe_load(open(args.config, "r"))
        # from thop import profile# 计算参数量
        # from thop import clever_format
        model = build_model(config)# 模型初始化
        # input1 = torch.randn(1,3,256,256)
        # input2 = torch.randn(1,1,256,256)
        # flops,params = profile(model,(input1,input2,))
        # # flops, params = clever_format([flops,params],'%.3f')
        # print('flops:%.2f M ,params:%.2f M'%(flops/1e6,params/1e6))



        optimizer = build_optimizer(model)
        # 数据集
        train_dataloader = build_train_data_loader(args, classname)
        test_dataloader = build_test_data_loader(args, classname)
        model.cuda()

        for epoch in range(const.NUM_EPOCHS):# 训练轮数
            train_one_epoch(args,train_dataloader, model, optimizer, epoch)
            if (epoch + 1) % const.EVAL_INTERVAL == 0:
                eval_once(args,test_dataloader, model,epoch,f)
            if (epoch + 1) %1 == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    os.path.join(checkpoint_dir, "%d.pt" % epoch),
                )
        f.close()



def evaluate(args):
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    for classname in const.MVTEC_CATEGORIES:
        args.category = classname
        args.cat = classname
        f = open(os.path.join(const.CHECKPOINT_DIR +'/' +args.category+ '/' + args.category+'_eval.txt'), 'w')
        # per_pixel_rocauc, img_roc_auc, pixel_pro
        max_per_pixel_rocauc = 0
        max_img_roc_auc = 0
        max_sum = 0
        max_reult = []
        max_reult_epoch = 0

        for epoch in range(0,const.NUM_EPOCHS,1):
            args.checkpoint = const.CHECKPOINT_DIR +'/' +args.category + '/'+ str(epoch)+'.pt'
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint["model_state_dict"])
            test_dataloader = build_test_data_loader(args, classname)
            model.cuda()
            result = eval_once_2(args,test_dataloader, model,epoch,f)
            if max_per_pixel_rocauc < result[0]:
                max_per_pixel_rocauc = result[0]
            if max_img_roc_auc < result[1]:
                max_img_roc_auc = result[1]
            if max_sum < result[0] + result[1]:
                max_sum = result[0] + result[1]
                max_reult = result
                max_reult_epoch = result[2]
        f.write("<<<<<<<<每个指标的最大值>>>>>>>>>>")
        f.write('\n')
        f.write('max pixel ROCAUC: %.4f     max image ROCAUC: %.4f' % (
            max_per_pixel_rocauc, max_img_roc_auc))
        f.write('\n')
        f.write("<<<<<<<<两个指标的最大值>>>>>>>>>>")
        f.write('\n')
        f.write("<<<<<<<<epoch: %.4f>>>>>>>>>>" % (max_reult_epoch))
        f.write('\n')
        f.write('max pixel ROCAUC: %.4f     max image ROCAUC: %.4f' % (
            max_reult[0], max_reult[1]))
        f.write('\n')
        f.close()





def parse_args():
    parser = argparse.ArgumentParser(description="Train FastFlow on MVTec-AD dataset")
    parser.add_argument(
        "-cfg", "--config", type=str, required=True, help="path to config file"
    )
    parser.add_argument("--data", type=str, required=True, help="path to mvtec folder")
    parser.add_argument(
        "-cat",
        "--category",
        type=str,
        choices=const.MVTEC_CATEGORIES,
        required=True,
        help="category name in mvtec",
    )
    parser.add_argument("--eval", action="store_true", help="run eval only")
    parser.add_argument(
        "-ckpt", "--checkpoint", type=str, help="path to load checkpoint"
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()#参数
    if args.eval:
        evaluate(args)
    else:
        train(args)
