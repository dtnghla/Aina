_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 1  # bs: total bs in all gpus
num_worker = 1  # Set back to 1 to avoid redundant data loading
mix_prob = 0
empty_cache = True  # Free GPU memory after each iteration
enable_amp = True  # Enable automatic mixed precision to reduce memory usage
evaluate = False  # Disable evaluation during training to avoid OOM errors
clip_grad = 1.0  # Reduced from 5.0 to 1.0 to prevent NaN losses
eval_epoch = 800  # Only evaluate at the end of training (set to total epochs)
save_path = "exp/pointgroup_optimized"  # Use a different experiment name to avoid overwriting logs

# Our dataset has semantic IDs 0 and 1
class_names = ["background", "pipe"]
num_classes = 2
segment_ignore_index = (-1, 0)  # -1 for ignored class, 0 for background

# model settings - reduced model size
model = dict(
    type="PG-v1m1",
    backbone=dict(
        type="SpUNet-v1m1",
        in_channels=6,  # xyz coordinates + rgb colors
        num_classes=0,
        channels=(32, 64, 128, 256, 128, 96, 96, 64),  # Reduced channel dimensions
        layers=(2, 2, 3, 4, 2, 2, 2, 2),  # Reduced layer counts
    ),
    backbone_out_channels=64,  # Reduced from 96 to 64
    semantic_num_classes=num_classes,
    semantic_ignore_index=-1,
    segment_ignore_index=segment_ignore_index,
    instance_ignore_index=-1,
    cluster_thresh=1.5,
    cluster_closed_points=200,
    cluster_propose_points=50,
    cluster_min_points=25,
)

# scheduler settings
epoch = 800
optimizer = dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=0.0001, nesterov=True)  # Reduced lr from 0.01 to 0.001
scheduler = dict(type="PolyLR")

# dataset settings
dataset_type = "LAZDataset"
data_root = "data"

data = dict(
    num_classes=num_classes,
    ignore_index=-1,
    names=class_names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(
                type="GridSample",
                grid_size=0.15,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                keys=("coord", "color", "segment", "instance"),
            ),
            dict(type="SphereCrop", point_max=25000, mode="random"),  # Further reduced from 30000 to 25000
            dict(type="NormalizeColor"),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "instance",
                    "instance_centroid",
                    "bbox",
                    "color",
                ),
                feat_keys=("color",),
            ),
        ],
        test_mode=False,
        loop=4,  # Reduced from 8 to 4 to speed up training
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="Copy",
                keys_dict={
                    "coord": "origin_coord",
                    "segment": "origin_segment",
                    "instance": "origin_instance",
                },
            ),
            dict(
                type="GridSample",
                grid_size=0.3,  # Further increased grid size for validation
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                keys=("coord", "color", "segment", "instance"),
            ),
            dict(type="SphereCrop", point_max=20000, mode="center"),  # Reduced from 30000 to 20000
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "instance",
                    "origin_coord",
                    "origin_segment",
                    "origin_instance",
                    "instance_centroid",
                    "bbox",
                    "color",
                ),
                feat_keys=("color",),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.3,  # Further increased grid size for testing
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                keys=("coord", "color", "segment", "instance"),
            ),
            dict(type="SphereCrop", point_max=20000, mode="center"),  # Reduced from 30000 to 20000
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "instance",
                    "instance_centroid",
                    "bbox",
                    "color",
                ),
                feat_keys=("color",),
            ),
        ],
        test_mode=True,
    ),
)

hooks = [
    dict(type="CheckpointLoader", keywords="module.", replacement="module."),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(
        type="InsSegEvaluator",
        segment_ignore_index=segment_ignore_index,
        instance_ignore_index=-1,
    ),
    dict(type="CheckpointSaver", save_freq=None),
] 