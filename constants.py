FEATURE_EXTRACTOR_CHECKPOINT_PATH = '/gpfsscratch/rech/ohv/ueu39kt/mvtec/deit_base_distilled_patch16_384-d0272ac0.pth'
CHECKPOINT_DIR = '/gpfsscratch/rech/ohv/ueu39kt/FastFlowGathierry/weights/CartetNewFastFlow_OriginalMain'
MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
    "carpet_color_augment",
]

BACKBONE_DEIT = "deit_base_distilled_patch16_384"
BACKBONE_CAIT = "cait_m48_448"
BACKBONE_RESNET18 = "resnet18"
BACKBONE_WIDE_RESNET50 = "wide_resnet50_2"

SUPPORTED_BACKBONES = [
    BACKBONE_DEIT,
    BACKBONE_CAIT,
    BACKBONE_RESNET18,
    BACKBONE_WIDE_RESNET50,
]

BATCH_SIZE = 32
NUM_EPOCHS = 500
LR = 1e-3
WEIGHT_DECAY = 1e-5

LOG_INTERVAL = 10
EVAL_INTERVAL = 10
CHECKPOINT_INTERVAL = 10
