import sklearn.metrics
import torch
import pytorch_lightning as pl
from torchvision.models import efficientnet_b0
import numpy as np
import os
import torchvision.transforms as transforms
from PIL import Image
from IPython.display import display

CONCEPT_MAP = [
    "Upper:FAU1",
    "Upper:FAU2",
    "Upper:FAU4",
    "Upper:FAU5",
    "Upper:FAU6",
    "Upper:FAU7",
    "Lower:FAU9",
    "Lower:FAU10",
    "Lower:FAU12",
    "Lower:FAU14",
    "Lower:FAU15",
    "Lower:FAU17",
    "Lower:FAU20",
    "Lower:FAU23",
    "Lower:FAU25",
    "Lower:FAU26",
    "Lower:FAU28",
    "Upper:FAU45",
]
LABEL_MAP = [
    "Neutral",
    "Anger",
    "Disgust",
    "Fear",
    "Happiness",
    "Sadness",
    "Surprise",
    "Other",
]

model_saved_path = os.path.join(os.getcwd(), "model", "eph40.pt")

pred_pkl_save_dir = os.path.dirname(model_saved_path)

device = torch.device("cpu")


def img_pre_processing(img):
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return transform(img)

CONFIG = dict(
    max_epochs=60,
    patience=15,
    batch_size=64,
    emb_size=16,
    extra_dims=0,
    concept_loss_weight=1,
    task_loss_weight=1,
    learning_rate=0.0001,
    weight_decay=4e-05,
    weight_loss=True,
    task_class_weights=[1.0, 10.691968865021419, 16.451397270448425, 19.515198237885464, 1.856195594104522, 2.2501047605744686, 5.6048711054879, 1.0683202102902343],
    c_extractor_arch="efficientnet_b0",
    x2c_used_pretrain = True,
    c2y_layers = [],
    optimizer="adam",
    embeding_activation="leakyrelu",
    bool=False,
    sampling_percent=1,
    shared_prob_gen=True,
    imbalance = [0.4699661339195531, 0.4072578351629601, 1.729288472145615, 1.0075462512171374, 1.8676905244856563, 1.9545020300931455, 2.604195498579649, 1.4049574726609966, 2.326968331876555, 1.178192543029449, 0.742508011409656, 3.446221583250966, 1.6664870399310234, 3.4034884755717716, 0.9276948848805953, 1.1970517716011013, 2.344508279824265, 1.446092243808394],
    architecture= "ConceptEmbeddingModel",
    x2c_used_pretrain_path = os.path.join(os.getcwd(), "model", "backbone.pt"),
)

def wrap_pretrained_model(c_extractor_arch, pretrain_model=True, x2c_used_pretrain=False, x2c_used_pretrain_path=''):
    def _result_x2c_fun(output_dim):

        if x2c_used_pretrain:
            model = torch.load(x2c_used_pretrain_path)
            model = torch.nn.Sequential(*list(model.children())[:-1])
            # print("Using the pretrained local network:", x2c_used_pretrain_path)
            if output_dim:
                model.classifier = torch.nn.Linear(
                    1280,
                    output_dim,
                )
            return model
        model = c_extractor_arch(pretrained=pretrain_model)

        return model
    return _result_x2c_fun

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

    model_cls = ConceptEmbeddingModel
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
    }

    if isinstance(config["c_extractor_arch"], str):
        if config["c_extractor_arch"] == "enet_b0_8_best_afew":
            c_extractor_arch = "enet_b0_8_best_afew"
        elif config["c_extractor_arch"] == "enet_b2_8_best":
            c_extractor_arch = "enet_b0_8_best_afew"
        elif config["c_extractor_arch"] == "efficientnet_b0":
            c_extractor_arch = efficientnet_b0
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
        task_loss_weight=config.get('task_loss_weight', 1.0),
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        c_extractor_arch=wrap_pretrained_model(c_extractor_arch, x2c_used_pretrain = config['x2c_used_pretrain'], x2c_used_pretrain_path=config['x2c_used_pretrain_path']),
        optimizer=config['optimizer'],
        top_k_accuracy=config.get('top_k_accuracy'),
        output_latent=output_latent,
        out_concept_generator=config.get('out_concept_generator', 1280),
        **extra_params,
    )

class ConceptBottleneckModel(pl.LightningModule):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        concept_loss_weight=0.01,
        task_loss_weight=1,

        extra_dims=0,
        bool=False,
        sigmoidal_prob=True,
        sigmoidal_extra_capacity=True,
        bottleneck_nonlinear=None,
        output_latent=False,

        x2c_model=None,
        c_extractor_arch=None,
        c2y_model=None,
        c2y_layers=None,

        optimizer="adam",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        weight_loss=None,
        task_class_weights=None,

        active_intervention_values=None,
        inactive_intervention_values=None,
        intervention_policy=None,

        top_k_accuracy=None,
        gpu=int(torch.cuda.is_available()),
    ):
        gpu = int(gpu)
        super().__init__()
        self.n_concepts = n_concepts
        self.intervention_policy = intervention_policy
        self.output_latent = output_latent
        if x2c_model is not None:
            self.x2c_model = x2c_model
        else:
            self.x2c_model = c_extractor_arch(
                output_dim=(n_concepts + extra_dims)
            )

        if c2y_model is not None:
            self.c2y_model = c2y_model
        else:
            units = [n_concepts + extra_dims] + (c2y_layers or []) + [n_tasks]
            layers = []
            for i in range(1, len(units)):
                layers.append(torch.nn.Linear(units[i-1], units[i]))
                if i != len(units) - 1:
                    layers.append(torch.nn.LeakyReLU())
            self.c2y_model = torch.nn.Sequential(*layers)

        init_fun = torch.cuda.FloatTensor if gpu else torch.FloatTensor
        if active_intervention_values is not None:
            self.active_intervention_values = init_fun(
                active_intervention_values
            )
        else:
            self.active_intervention_values = init_fun(
                [1 for _ in range(n_concepts)]
            ) * (
                5.0 if not sigmoidal_prob else 1.0
            )
        if inactive_intervention_values is not None:
            self.inactive_intervention_values = init_fun(
                inactive_intervention_values
            )
        else:
            self.inactive_intervention_values = init_fun(
                [1 for _ in range(n_concepts)]
            ) * (
                -5.0 if not sigmoidal_prob else 0.0
            )

        self.sig = torch.nn.Sigmoid()
        if sigmoidal_extra_capacity:
            bottleneck_nonlinear = "sigmoid"
        if bottleneck_nonlinear == "sigmoid":
            self.bottleneck_nonlin = torch.nn.Sigmoid()
        elif bottleneck_nonlinear == "leakyrelu":
            self.bottleneck_nonlin = torch.nn.LeakyReLU()
        elif bottleneck_nonlinear == "relu":
            self.bottleneck_nonlin = torch.nn.ReLU()
        elif (bottleneck_nonlinear is None) or (
            bottleneck_nonlinear == "identity"
        ):
            self.bottleneck_nonlin = lambda x: x
        else:
            raise ValueError(
                f"Unsupported nonlinearity '{bottleneck_nonlinear}'"
            )

        self.loss_concept = torch.nn.BCELoss(weight=weight_loss)
        self.loss_task = (
            torch.nn.CrossEntropyLoss(weight=task_class_weights)
            if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(
                weight=task_class_weights
            )
        )
        self.bool = bool
        self.concept_loss_weight = concept_loss_weight
        self.task_loss_weight = task_loss_weight
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.extra_dims = extra_dims
        self.top_k_accuracy = top_k_accuracy
        self.n_tasks = n_tasks
        self.sigmoidal_prob = sigmoidal_prob
        self.sigmoidal_extra_capacity = sigmoidal_extra_capacity

    def _unpack_batch(self, batch):
        x = batch[0]
        if isinstance(batch[1], list):
            y, c = batch[1]
        else:
            y, c = batch[1], batch[2]
        return x, y, c

    def _standardize_indices(self, intervention_idxs, batch_size):
        if isinstance(intervention_idxs, list):
            intervention_idxs = np.array(intervention_idxs)
        if isinstance(intervention_idxs, np.ndarray):
            intervention_idxs = torch.IntTensor(intervention_idxs)

        if intervention_idxs is None or (
            isinstance(intervention_idxs, torch.Tensor) and
            ((len(intervention_idxs) == 0) or intervention_idxs.shape[-1] == 0)
        ):
            return None
        if not isinstance(intervention_idxs, torch.Tensor):
            raise ValueError(
                f'Unsupported intervention indices {intervention_idxs}'
            )
        if len(intervention_idxs.shape) == 1:
            intervention_idxs = torch.tile(
                torch.unsqueeze(intervention_idxs, 0),
                (batch_size, 1),
            )
        elif len(intervention_idxs.shape) == 2:
            assert intervention_idxs.shape[0] == batch_size, (
                f'Expected intervention indices to have batch size {batch_size} '
                f'but got intervention indices with shape {intervention_idxs.shape}.'
            )
        else:
            raise ValueError(
                f'Intervention indices should have 1 or 2 dimensions. Instead we got '
                f'indices with shape {intervention_idxs.shape}.'
            )
        if intervention_idxs.shape[-1] == self.n_concepts:
            elems = torch.unique(intervention_idxs)
            if len(elems) == 1:
                is_binary = (0 in elems) or (1 in elems)
            elif len(elems) == 2:
                is_binary = (0 in elems) and (1 in elems)
            else:
                is_binary = False
        else:
            is_binary = False
        if not is_binary:
            intervention_idxs = intervention_idxs.to(dtype=torch.long)
            result = torch.zeros(
                (batch_size, self.n_concepts),
                dtype=torch.bool,
                device=intervention_idxs.device,
            )
            result[:, intervention_idxs] = 1
            intervention_idxs = result
        assert intervention_idxs.shape[-1] == self.n_concepts, (
                f'Unsupported intervention indices with shape {intervention_idxs.shape}.'
            )
        if isinstance(intervention_idxs, np.ndarray):
            intervention_idxs = torch.BoolTensor(intervention_idxs)
        intervention_idxs = intervention_idxs.to(dtype=torch.bool)
        return intervention_idxs

    def _concept_intervention(
        self,
        c_pred,
        intervention_idxs=None,
        c_true=None,
    ):
        if (c_true is None) or (intervention_idxs is None):
            return c_pred
        c_pred_copy = c_pred.clone()
        intervention_idxs = self._standardize_indices(
            intervention_idxs=intervention_idxs,
            batch_size=c_pred.shape[0],
        )
        intervention_idxs = intervention_idxs.to(c_pred.device)
        if self.sigmoidal_prob:
            c_pred_copy[intervention_idxs] = c_true[intervention_idxs]
        else:
            batched_active_intervention_values =  torch.tile(
                torch.unsqueeze(self.active_intervention_values, 0),
                (c_pred.shape[0], 1),
            )

            batched_inactive_intervention_values =  torch.tile(
                torch.unsqueeze(self.inactive_intervention_values, 0),
                (c_pred.shape[0], 1),
            )

            c_pred_copy[intervention_idxs] = (
                (
                    c_true[intervention_idxs] *
                    batched_active_intervention_values[intervention_idxs]
                ) +
                (
                    (c_true[intervention_idxs] - 1) *
                    -batched_inactive_intervention_values[intervention_idxs]
                )
            )

        return c_pred_copy

    def _forward(
        self,
        x,
        intervention_idxs=None,
        c=None,
        y=None,
        train=False,
        latent=None,
    ):
        if latent is None:
            latent = self.x2c_model(x)
        if self.sigmoidal_prob or self.bool:
            if self.extra_dims:
                c_pred_probs = self.sig(latent[:, :-self.extra_dims])
                c_others = self.bottleneck_nonlin(latent[:,-self.extra_dims:])
                c_pred =  torch.cat([c_pred_probs, c_others], axis=-1)
                c_sem = c_pred_probs
            else:
                c_pred = self.sig(latent)
                c_sem = c_pred
        else:
            c_pred = latent
            if self.extra_dims:
                c_sem = self.sig(latent[:, :-self.extra_dims])
            else:
                c_sem = self.sig(latent)
        if (intervention_idxs is None) and (c is not None) and (
            self.intervention_policy is not None
        ):
            intervention_idxs, c_int = self.intervention_policy(
                x=x,
                c=c,
                pred_c=c_sem,
                y=y,
            )
        else:
            c_int = c
        c_pred = self._concept_intervention(
            c_pred=c_pred,
            intervention_idxs=intervention_idxs,
            c_true=c_int,
        )
        if self.bool:
            y = self.c2y_model((c_pred > 0.5).float())
        else:
            y = self.c2y_model(c_pred)
        if self.output_latent:
            return c_sem, c_pred, y, latent
        return c_sem, c_pred, y

    def forward(self, x, c=None, y=None, latent=None, intervention_idxs=None):
        return self._forward(
            x,
            train=False,
            c=c,
            y=y,
            intervention_idxs=intervention_idxs,
            latent=latent,
        )


class ConceptEmbeddingModel(ConceptBottleneckModel):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        emb_size=16,
        training_intervention_prob=0.25,
        embeding_activation="leakyrelu",
        shared_prob_gen=True,
        concept_loss_weight=1,
        task_loss_weight=1,

        c2y_model=None,
        c2y_layers=None,
        y_activation="sigmoid",
        c_extractor_arch=None,
        output_latent=False,

        optimizer="adam",
        momentum=0.9,
        learning_rate=0.0001,
        weight_decay=4e-05,
        weight_loss=None,
        task_class_weights=None,

        active_intervention_values=None,
        inactive_intervention_values=None,
        intervention_policy=None,

        top_k_accuracy=2,
        out_concept_generator=1280,
        gpu=int(torch.cuda.is_available()),
    ):
        pl.LightningModule.__init__(self)
        self.n_concepts = n_concepts
        self.intervention_policy = intervention_policy
        self.pre_concept_model = c_extractor_arch(output_dim=None)
        self.training_intervention_prob = training_intervention_prob
        self.output_latent = output_latent
        if self.training_intervention_prob != 0:
            self.ones = torch.ones(n_concepts)

        if active_intervention_values is not None:
            self.active_intervention_values = torch.tensor(
                active_intervention_values
            )
        else:
            self.active_intervention_values = torch.ones(n_concepts)
        if inactive_intervention_values is not None:
            self.inactive_intervention_values = torch.tensor(
                inactive_intervention_values
            )
        else:
            self.inactive_intervention_values = torch.ones(n_concepts)
        self.task_loss_weight = task_loss_weight
        self.concept_context_generators = torch.nn.ModuleList()
        self.concept_prob_generators = torch.nn.ModuleList()
        self.shared_prob_gen = shared_prob_gen
        self.top_k_accuracy = top_k_accuracy
        for i in range(n_concepts):
            if embeding_activation is None:
                self.concept_context_generators.append(
                    torch.nn.Linear(
                        list(
                            self.pre_concept_model.modules()
                        )[-1].out_features,
                        2 * emb_size,
                    )
                )
            elif embeding_activation == "sigmoid":
                self.concept_context_generators.append(
                    torch.nn.Sequential(*[
                        torch.nn.Linear(
                            out_concept_generator,
                            2 * emb_size,
                        ),
                        torch.nn.Sigmoid(),
                    ])
                )
            elif embeding_activation == "leakyrelu":
                self.concept_context_generators.append(
                    torch.nn.Sequential(*[
                        torch.nn.Linear(
                            out_concept_generator,
                            (2 * emb_size),
                        ),
                        torch.nn.LeakyReLU(),
                    ])
                )
            elif embeding_activation == "relu":
                self.concept_context_generators.append(
                    torch.nn.Sequential(*[
                        torch.nn.Linear(
                            out_concept_generator,
                            2 * emb_size,
                        ),
                        torch.nn.ReLU(),
                    ])
                )
            if self.shared_prob_gen and (
                len(self.concept_prob_generators) == 0
            ):
                self.concept_prob_generators.append(
                    torch.nn.Linear(
                        2 * emb_size,
                        1,
                    )
                )
            elif not self.shared_prob_gen:
                self.concept_prob_generators.append(
                    torch.nn.Linear(
                        2 * emb_size,
                        1,
                    )
                )
        if c2y_model is None:
            units = [
                n_concepts * emb_size
            ] + (c2y_layers or []) + [n_tasks]
            layers = []
            for i in range(1, len(units)):
                layers.append(torch.nn.Linear(units[i-1], units[i]))
                if i != len(units) - 1:
                    layers.append(torch.nn.LeakyReLU())
            self.c2y_model = torch.nn.Sequential(*layers)
        else:
            self.c2y_model = c2y_model

        if y_activation=="sigmoid":
            self.sig = torch.nn.Sigmoid()
        if y_activation=="softmax":
            self.sig = torch.nn.Softmax(dim=1)

        self.loss_concept = torch.nn.BCELoss(weight=weight_loss)
        self.loss_task = (
            torch.nn.CrossEntropyLoss(weight=task_class_weights)
            if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(
                weight=task_class_weights
            )
        )
        self.concept_loss_weight = concept_loss_weight
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.n_tasks = n_tasks
        self.emb_size = emb_size

    def _after_interventions(
        self,
        prob,
        intervention_idxs=None,
        c_true=None,
        train=False,
    ):
        if train and (self.training_intervention_prob != 0) and (
            (c_true is not None) and
            (intervention_idxs is None)
        ):
            mask = torch.bernoulli(
                self.ones * self.training_intervention_prob,
            )
            intervention_idxs = torch.tile(
                mask,
                (c_true.shape[0], 1),
            )
        if (c_true is None) or (intervention_idxs is None):
            return prob
        intervention_idxs = intervention_idxs.to(prob.device)
        intervention_idxs = intervention_idxs.to(dtype=torch.int32)
        return prob * (1 - intervention_idxs) + intervention_idxs * c_true

    def _forward(
        self,
        x,
        intervention_idxs=None,
        c=None,
        y=None,
        train=False,
        latent=None,
    ):
        if latent is None:
            pre_c = self.pre_concept_model(x)
            contexts = []
            c_sem = []
            for i, context_gen in enumerate(self.concept_context_generators):
                if self.shared_prob_gen:
                    prob_gen = self.concept_prob_generators[0]
                else:
                    prob_gen = self.concept_prob_generators[i]
                context = context_gen(pre_c)
                prob = prob_gen(context)
                contexts.append(torch.unsqueeze(context, dim=1))
                c_sem.append(self.sig(prob))
            c_sem = torch.cat(c_sem, axis=-1)
            contexts = torch.cat(contexts, axis=1)
            latent = contexts, c_sem
        else:
            contexts, c_sem = latent
        if (intervention_idxs is None) and (c is not None) and (
            self.intervention_policy is not None
        ):
            intervention_idxs, c_int = self.intervention_policy(
                x=x,
                c=c,
                pred_c=c_sem,
                y=y,
            )

        else:
            c_int = c
        intervention_idxs = self._standardize_indices(
            intervention_idxs=intervention_idxs,
            batch_size=x.shape[0],
        )
        probs = self._after_interventions(
            c_sem,
            intervention_idxs=intervention_idxs,
            c_true=c_int,
            train=train,
        )
        c_pred = (
            contexts[:, :, :self.emb_size] * torch.unsqueeze(probs, dim=-1) +
            contexts[:, :, self.emb_size:] * (1 - torch.unsqueeze(probs, dim=-1))
        )
        c_pred = c_pred.view((-1, self.emb_size * self.n_concepts))
        y = self.c2y_model(c_pred)
        if self.output_latent:
            return c_sem, c_pred, y, latent
        return c_sem, c_pred, y