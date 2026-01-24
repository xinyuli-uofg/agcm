import os, datetime, shutil, pickle, cv2, torch
from glob import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
import torchvision.transforms as T
import sys
sys.path.append("/home/{your_user_name}/work/cem")
from xtools.xy_draw_au_map import (
    visualize_patch_based_heatmap,
    visualize_weighted_heatmaps,
)
import train.utils as utils
import models.ag_cem_ROI as models_ag_cem_ROI

def load_config():
    config = dict(
        concept_selection='12_au_full',
        reg_concept='huber',
        cv=1,
        wandb_enable=False,
        wandb_project_name="",
        wandb_run_name_ext="",
        resume_from_pretrain="./ckpt/backbone.pth",
        max_epochs=30,
        frozen_epochs=0,
        debug=False,
        patience=15,
        batch_size=128,
        emb_size=16,
        extra_dims=0,
        concept_loss_weight=1,
        task_loss_weight=1,
        learning_rate=0.00001,
        weight_decay=4e-05,
        weight_loss=False,
        task_class_weights=[1.0, 10.399988319803773, 16.23179290857716, 19.607905747632678, 1.8556467915720152,
                            2.225347712532647, 5.610554505356018, 1.0590043828089226],
        c_extractor_arch="efficientnet_b0",
        x2c_used_pretrain=True,
        x2c_used_pretrain_path="./ckpt/backbone.pth",
        c2y_layers=[],
        optimizer="adam",
        embeding_activation="leakyrelu",
        y_activation="sigmoid",
        unfreeze_learning_rate=0.00001,
        bool=False,
        early_stopping=False,
        early_stopping_monitor="val_loss",
        early_stopping_mode="min",
        early_stopping_delta=0.0,
        sampling_percent=1,
        momentum=0.9,
        sigmoidal_prob=False,
        training_intervention_prob=0.25,
        intervention_freq=4,
        shared_prob_gen=True,
        check_val_every_n_epoch=10,
        dataset_name="AffectNet",
    )

    config['num_workers'] = 8
    config["architecture"] = "Attention_ConceptEmbeddingModel_ROI"
    config["extra_name"] = f""
    config["sigmoidal_prob"] = True
    config['emb_size'] = config['emb_size']
    return config


def construct_model(
    n_concepts,
    n_tasks,
    config,
    c2y_model=None,
    x2c_model=None,
    imbalance=None,
    task_class_weights=None,
    intervention_policy=None,
    active_intervention_values=None,
    inactive_intervention_values=None,
    output_latent=False,
):
    if config["architecture"] in ["Attention_ConceptEmbeddingModel_ROI"]:
        model_cls = models_ag_cem_ROI.Attention_ConceptEmbeddingModel_ROI
        c_extractor_arch = config["c_extractor_arch"]
        extra_params = {
            "emb_size": config["emb_size"],
            "shared_prob_gen": config["shared_prob_gen"],
            "intervention_policy": intervention_policy,
            "training_intervention_prob": config.get(
                'training_intervention_prob',
                0.0,
            ),
            "embeding_activation": config.get("embeding_activation", None),
            "c2y_model": c2y_model,
            "c2y_layers": config.get("c2y_layers", []),
            "y_activation": config.get("y_activation", "sigmoid"),
            "architecture": config.get("architecture", None),
            "reg_concept": config.get("reg_concept", None),
        }
        return model_cls(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            weight_loss=(
                torch.FloatTensor(imbalance)
                if config['weight_loss'] and (imbalance is not None)
                else None
            ),
            task_class_weights=(
                torch.FloatTensor(task_class_weights)
                if (task_class_weights is not None)
                else None
            ),
            concept_loss_weight=config['concept_loss_weight'],
            map_loss_weight=config.get('map_loss_weight', 0),
            task_loss_weight=config.get('task_loss_weight', 1.0),
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            c_extractor_arch=utils.wrap_pretrained_model(c_extractor_arch,
                                                         x2c_used_pretrain=config['x2c_used_pretrain'],
                                                         x2c_used_pretrain_path=config['x2c_used_pretrain_path'],
                                                         architecture=config['architecture']),
            optimizer=config['optimizer'],
            top_k_accuracy=config.get('top_k_accuracy'),
            output_latent=output_latent,
            out_concept_generator=config.get('out_concept_generator', 1280),
            **extra_params,
        )

    else:
        raise ValueError(f'Invalid architecture "{config["architecture"]}"')

    if isinstance(config["c_extractor_arch"], str):
        if config["c_extractor_arch"] == "resnet18":
            c_extractor_arch = resnet18
        elif config["c_extractor_arch"] == "resnet34":
            c_extractor_arch = resnet34
        elif config["c_extractor_arch"] == "resnet50":
            c_extractor_arch = resnet50
        elif config["c_extractor_arch"] == "densenet121":
            c_extractor_arch = densenet121
        elif config["c_extractor_arch"] == "efficientnet_b4":
            c_extractor_arch = efficientnet_b4
        elif config["c_extractor_arch"] == "efficientnet_b0":
            c_extractor_arch = efficientnet_b0
        elif config["c_extractor_arch"] == "enet_b0_8_best_afew":
            c_extractor_arch = "enet_b0_8_best_afew"
        elif config["c_extractor_arch"] == "enet_b2_8_best":
            c_extractor_arch = "enet_b0_8_best_afew"
        else:
            raise ValueError(f'Invalid model_to_use "{config["model_to_use"]}"')
    else:
        c_extractor_arch = config["c_extractor_arch"]

    return model_cls(
        n_concepts=n_concepts,
        n_tasks=n_tasks,
        weight_loss=(
            torch.FloatTensor(imbalance)
            if config['weight_loss'] and (imbalance is not None)
            else None
        ),
        task_class_weights=(
            torch.FloatTensor(task_class_weights)
            if (task_class_weights is not None)
            else None
        ),
        concept_loss_weight=config['concept_loss_weight'],
        map_loss_weight=config.get('map_loss_weight', 0),
        task_loss_weight=config.get('task_loss_weight', 1.0),
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        c_extractor_arch=utils.wrap_pretrained_model(c_extractor_arch, x2c_used_pretrain = config['x2c_used_pretrain'], x2c_used_pretrain_path=config['x2c_used_pretrain_path'], architecture=config['architecture']),
        optimizer=config['optimizer'],
        top_k_accuracy=config.get('top_k_accuracy'),
        output_latent=output_latent,
        out_concept_generator=config.get('out_concept_generator', 1280),
        **extra_params,
    )

CONCEPT_MAP = [
    "Upper:FAU1","Upper:FAU2","Upper:FAU4","Upper:FAU5","Upper:FAU6","Upper:FAU7",
    "Lower:FAU9","Lower:FAU10","Lower:FAU12","Lower:FAU14","Lower:FAU15","Lower:FAU17",
    "Lower:FAU20","Lower:FAU23","Lower:FAU25","Lower:FAU26","Lower:FAU28","Upper:FAU45",
]
LABEL_MAP = ["Neutral","Happiness","Sadness","Surprise","Fear","Disgust","Anger","Contempt"]

def au_score_table(scores, concept_map, n_rows=None):
    scores = np.asarray(scores).flatten()
    order  = np.argsort(scores)[::-1]
    if n_rows is not None:
        order = order[:n_rows]
    w = max(len(concept_map[i]) for i in order) + 2
    lines = [f"{'AU name'.ljust(w)}| score", '-'*(w+7)]
    lines += [f"{concept_map[i].ljust(w)}| {scores[i]:.4f}" for i in order]
    lines.append('-'*(w+7))
    return lines

config = load_config()
config['x2c_used_pretrain_path'] = "./ckpt/backbone.pth"
model_saved_path = "./ckpt/agcm_affectnet_100e.ckpt"
pred_pkl_save_dir = os.path.dirname(model_saved_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = construct_model(
    18,
    8,
    config,
    imbalance=[0.4699661339195531, 0.4072578351629601, 1.729288472145615, 1.0075462512171374, 1.8676905244856563, 1.9545020300931455, 2.604195498579649, 1.4049574726609966, 2.326968331876555, 1.178192543029449, 0.742508011409656, 3.446221583250966, 1.6664870399310234, 3.4034884755717716, 0.9276948848805953, 1.1970517716011013, 2.344508279824265, 1.446092243808394],
    task_class_weights=config.get('task_class_weights', None),
)
if model_saved_path.endswith('.pt'):
    print("Resume Training from pt:", model_saved_path)
    model.load_state_dict(torch.load(model_saved_path))
elif model_saved_path.endswith('.ckpt'):
    print("Resume Training from ckpt:", model_saved_path)
    checkpoint = torch.load(model_saved_path, map_location=torch.device('cpu'))
    print(checkpoint.keys())
    model.load_state_dict(checkpoint['state_dict'])

model.freeze()
model.to(device)

with open(VAL_PKL, 'rb') as f:
    val_aff_data = pickle.load(f)
path2meta = {d['img_path']: d for d in val_aff_data}

tfm = T.Compose([
    T.Resize([224,224]),
    T.ToTensor(),
    T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])

def process_image(img_path, k=5):
    stem = os.path.splitext(os.path.basename(img_path))[0]

    meta      = path2meta.get(img_path, None)
    gt_label  = LABEL_MAP[meta['class_label']] if meta else 'unknown'

    out_dir   = os.path.join(BASE_OUTPUT_DIR, f"{stem}_{gt_label.lower()}")
    img_dir   = os.path.join(out_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    log_path  = os.path.join(out_dir, 'results.txt')

    required = [
        os.path.join(img_dir, f'{stem}.jpg'),
        os.path.join(img_dir, f'{stem}_au_GT.jpg'),
        os.path.join(img_dir, f'{stem}_au_pred.jpg'),
        os.path.join(img_dir, f'{stem}_norm_pred.jpg'),
        os.path.join(img_dir, f'{stem}_norm_GT.jpg'),
        os.path.join(out_dir, 'results.txt'),
    ]
    if all(os.path.isfile(f) for f in required):
        return

    def log(*args):
        msg = ' '.join(map(str, args))
        print(msg)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    if not os.path.isfile(log_path):
        log("#"*60)
        log(f"Image : {img_path}")
        log(f"GT    : {gt_label}")
        log("#"*60)
    with torch.no_grad():
        inp = tfm(Image.open(img_path).convert('RGB')).unsqueeze(0).repeat(128,1,1,1).to(device)
        c_sem, _, y_pred, spatial_map, _ = model(inp)
    pred_idx = y_pred[0].argmax().item()
    pred_cls = LABEL_MAP[pred_idx]
    scores_np  = c_sem[0].cpu().numpy()
    log("Pred class :", pred_cls)
    log("Pred AU table:")
    for l in au_score_table(scores_np, CONCEPT_MAP, n_rows=None):
        log(l)

    save_gt      = os.path.join(img_dir, f'{stem}_au_GT.jpg')
    save_pred    = os.path.join(img_dir, f'{stem}_au_pred.jpg')
    save_norm    = os.path.join(img_dir, f'{stem}_norm_pred.jpg')
    save_norm_gt = os.path.join(img_dir, f'{stem}_norm_GT.jpg')

    att_map    = spatial_map[0].cpu().numpy()
    raw_cv     = cv2.imread(img_path)
    visualize_patch_based_heatmap(raw_cv, att_map, save_pred, normalize=True, separate=True)
    scores_np[scores_np < 0.5] = 0.0
    visualize_weighted_heatmaps(raw_cv, att_map, scores_np, save_norm, normalize=True)
    shutil.copy(img_path, os.path.join(img_dir, os.path.basename(img_path)))
    log("Images saved to", img_dir, '\n')

