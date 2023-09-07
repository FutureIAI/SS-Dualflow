import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from perlin import rand_perlin_2d_np

class MVTecDRAEMTestDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None):
        self.root_dir = root_dir
        self.images = sorted(glob.glob(root_dir+"/*/*.png"))
        self.resize_shape=resize_shape

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+"_mask.png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,'mask': mask, 'idx': idx}

        return sample



class MVTecDRAEMTrainDataset(Dataset):

    def __init__(self, root_dir, anomaly_source_path, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape=resize_shape

        self.image_paths = sorted(glob.glob(root_dir+"/*.png"))

        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])


    def __len__(self):
        return len(self.image_paths)


    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def augment_image(self, image, anomaly_source_path):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

        anomaly_img_augmented = aug(image=anomaly_source_img)
        # perlin noise mask柏林噪声掩模
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)
        # ----------------
        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)

        # 生成纹理和结构异常
        no_anomaly = torch.rand(1).numpy()[0]
        # no_anomaly = 0.1
        if no_anomaly > 0.5:# 大于0.5时,返回的是原图(无异常)
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:# 小于0.5时,返回的是有缺陷的图像
            augmented_image = augmented_image.astype(np.float32)
            # cv2.imwrite('parts11_异常图像----.jpg', augmented_image)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1 - msk) * image
            # cv2.imwrite('parts00_原图.jpg', image*255)
            # cv2.imwrite('parts11_异常图像.jpg', augmented_image*255)
            # cv2.imwrite('parts22_异常mask.jpg', msk*255)
            # 获取图像的遮罩
            target_foreground_mask = self.target_foreground_mask((image*255).astype(np.uint8))
            # target_foreground_mask = self.target_foreground_mask((image).astype(np.uint8))
            target_foreground_mask = target_foreground_mask.astype(np.float32)
            target_foreground_mask = np.expand_dims(target_foreground_mask, axis=2)
            # cv2.imwrite('parts33-最后的遮罩.jpg', target_foreground_mask * 255)
            # 将遮罩外的异常还原
            #还原图像
            # cv2.imwrite('parts44_hhhh.jpg', augmented_image)
            augmented_image = target_foreground_mask * augmented_image + (1 - target_foreground_mask) * image
            msk = target_foreground_mask * msk
            # cv2.imwrite('parts44_遮罩完的异常.jpg', augmented_image*255 )
            # cv2.imwrite('parts55_遮罩完的异常标识.jpg', msk * 255)

            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly=0.0
            return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)

    def transform_image(self, image_path, anomaly_source_path):
        image = cv2.imread(image_path)
        # cv2.imshow("1", image)
        image = cv2.resize(image
                           , dsize=(self.resize_shape[1], self.resize_shape[0]))
        # cv2.imshow("2",image)
        # do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        # if do_aug_orig:
        #     image = self.rot(image=image)# 将图像旋转
        # cv2.imshow("3", image)
        # cv2.waitKey(0)
        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0# 归一化到0-1
        # image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32)
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)# 得到缺陷图像，缺陷位置，是否有缺陷

        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        return image, augmented_image, anomaly_mask, has_anomaly

    def target_foreground_mask(self,img):
        # cv2.imwrite('parts0_原图.jpg', img)
        ## target foreground mask 目标前景遮罩
        # target_foreground_mask_1 = self.generate_target_foreground_mask(img=img)# memSeg方法
        # # -------------------
        # cv2.imwrite('memSeg方法生成前景遮罩.png', target_foreground_mask_1*255)
        # ttt = np.stack((target_foreground_mask_1 * 255,) * 3, axis=-1)
        # ttt = np.array(ttt, dtype='uint8')
        # target_foreground_mask_1 = np.array(ttt[:, :, 0], dtype=int)

        ## 获取边缘信息
        texture = cv2.Canny(img, 50, 100, L2gradient=True)
        # texture = cv2.Canny(img, 50, 100, L2gradient=True)
        # cv2.imwrite('parts2_纹理.jpg', texture)
        # 将图像边缘信息记录下来,因为canny算子预测不到图像边缘边界  FPC数据集用得到
        # texture[:, 0] = target_foreground_mask[:, 0]
        # texture[:, 255] = target_foreground_mask[:, 255]
        # texture[0, :] = target_foreground_mask[0,:]
        # texture[255, :] = target_foreground_mask[255, 0]
        # texture = np.stack((texture,) * 3, axis=-1)
        # cv2.imwrite('parts2.jpg', texture)
        ## 使用闭运算连接中断的图像前景,迭代运算三次
        # 开：先腐蚀，再膨胀
        kernel = np.ones((5, 5), np.uint8)
        texture = cv2.morphologyEx(texture, cv2.MORPH_CLOSE, kernel=kernel, iterations=5)
        # cv2.imwrite('parts3_纹理连线.jpg', texture)
        ## 获取最大轮廓进行填充
        # threshold
        thresh = cv2.threshold(texture, 128, 255, cv2.THRESH_BINARY)[1]
        # get the (largest) contour获得（最大）轮廓
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        big_contour = max(contours, key=cv2.contourArea)
        # draw white filled contour on black background在黑色背景上绘制白色填充轮廓
        target_foreground_mask_2 = np.zeros_like(texture)
        cv2.drawContours(target_foreground_mask_2, [big_contour], 0, (255, 255, 255), cv2.FILLED)
        # cv2.imwrite('parts4_纹理填充.jpg', target_foreground_mask_2)
        target_foreground_mask_2 = np.array(target_foreground_mask_2, dtype=int)

        # 如果纹理信息提取的不好,则用原来的
        # cv2.imwrite('纹理方法生成前景遮罩.png', target_foreground_mask_2)
        target_foreground_mask = target_foreground_mask_2/255
        # if (target_foreground_mask_2 / 255).sum() < 1500:  # 如果用边缘扣出的mask不好,则用原来的
        #     target_foreground_mask = target_foreground_mask_1
        # 合并两个mask
        # target_foreground_mask = (target_foreground_mask_2+target_foreground_mask_1)
        # target_foreground_mask = np.where(target_foreground_mask > 255, 255, target_foreground_mask)
        # cv2.imwrite('parts6_最后遮罩.jpg', target_foreground_mask)
        return target_foreground_mask

    def generate_target_foreground_mask(self, img: np.ndarray) -> np.ndarray:
        # convert RGB into GRAY scale
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # generate binary mask of gray scale image
        _, target_background_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        target_background_mask = target_background_mask.astype(np.bool).astype(np.int)

        # invert mask for foreground mask
        target_foreground_mask = -(target_background_mask - 1)

        return target_foreground_mask

    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(self.image_paths[idx],
                                                                           self.anomaly_source_paths[anomaly_source_idx])
        sample = {'image': image, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}

        return sample


