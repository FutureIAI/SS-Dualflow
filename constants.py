
CHECKPOINT_DIR = "/home/cgz/MZL/SS-Dualflow/result"

MVTEC_CATEGORIES = [
        "bottle",
        "cable",# (quan 0)
        "transistor",
        "wood",# (quan 1)
        "capsule",
        "carpet",# (quan 1)
        "grid",# (quan 1)
        "hazelnut",
        "leather",# (quan 1)
        "metal_nut",#
        "pill",
        "screw",
        "tile",# (quan 1)
        "toothbrush",# (xuyaoquan meiquan)
        "zipper",
]
BACKBONE_DEIT = "deit_base_distilled_patch16_384"
BACKBONE_CAIT = "cait_m48_448"
BACKBONE_RESNET18 = "resnet18"
BACKBONE_WIDE_RESNET50 = "wide_resnet50_2"

SUPPORTED_BACKBONES = [
    BACKBONE_RESNET18,
]

BATCH_SIZE = 32
NUM_EPOCHS = 1000
LR = 1e-3
WEIGHT_DECAY = 1e-5

LOG_INTERVAL = 10
EVAL_INTERVAL = 10
CHECKPOINT_INTERVAL = 10

#  cv2.Canny()的参数 threshold1 和 threshold2
threshold1 = 50
threshold2 =100

# 测试起始epoch
test_begin_epoch = 0

flow_wai = 4
flow_li = 4