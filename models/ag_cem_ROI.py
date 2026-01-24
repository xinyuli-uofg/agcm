
import torch
import pytorch_lightning as pl
from models.cbm_ROI import ConceptBottleneckModel
import train.utils as utils

from torchvision.models import resnet50

class LambdaLayer(torch.nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)


def cosine_similarity_attention_loss(predicted_attention, ground_truth_attention):
    predicted_attention_norm = torch.nn.functional.normalize(predicted_attention.view(predicted_attention.size(0), -1), dim=1)
    ground_truth_attention_norm = torch.nn.functional.normalize(ground_truth_attention.view(ground_truth_attention.size(0), -1), dim=1)
    cosine_sim = torch.sum(predicted_attention_norm * ground_truth_attention_norm, dim=1)
    loss = 1 - cosine_sim.mean()
    return loss

class StaticPatchSpatialAttention(torch.nn.Module):
    def __init__(self):
        super(StaticPatchSpatialAttention, self).__init__()
        self.query = torch.nn.Parameter(torch.randn(1, 196, 1))

    def forward(self, x):
        patch_embeddings = x[:, 1:, :]
        attention_scores = torch.softmax(self.query, dim=1)
        attended_patches = patch_embeddings * attention_scores
        return torch.cat((x[:, :1, :], attended_patches), dim=1), attention_scores


class _ConvolutionalPatchSpatialAttention(torch.nn.Module):
    def __init__(self, num_patches, channels, patch_size=14):
        super(_ConvolutionalPatchSpatialAttention, self).__init__()
        self.conv_query = torch.nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=3, padding=1)
        self.patch_size = patch_size

    def forward(self, x):
        cls_token = x[:, 0:1, :]
        x_patches = x[:, 1:, :]
        x_patches = x_patches.view(x.shape[0], -1, self.patch_size, self.patch_size)
        query = self.conv_query(x_patches)
        query = query.view(x.shape[0], -1).unsqueeze(-1)
        attention_scores = torch.softmax(query, dim=1)
        attention_scores = attention_scores.view(x.shape[0], 1, self.patch_size, self.patch_size)
        attended_patches = x_patches * attention_scores
        attended_patches = attended_patches.view(x.shape[0], -1, x.shape[-1])
        return torch.cat((cls_token, attended_patches), dim=1), attention_scores


class ConvolutionalPatchSpatialAttention(torch.nn.Module):

    def __init__(self, num_patches, channels, patch_size=14, dropout_rate=0.1):
        super(ConvolutionalPatchSpatialAttention, self).__init__()
        self.conv_query = torch.nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=3, padding=1)
        self.batch_norm = torch.nn.BatchNorm2d(1)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = torch.nn.LeakyReLU(negative_slope=0.01)
        self.patch_size = patch_size

    def forward(self, x):
        cls_token = x[:, 0:1, :]
        x_patches = x[:, 1:, :]
        x_patches = x_patches.view(x.shape[0], -1, self.patch_size, self.patch_size)
        query = self.conv_query(x_patches)
        query = self.batch_norm(query)
        query = query.view(x.shape[0], -1).unsqueeze(-1)
        attention_scores = torch.softmax(query, dim=1)
        attention_scores = self.dropout(attention_scores)
        attention_scores = attention_scores.view(x.shape[0], 1, self.patch_size, self.patch_size)
        attended_patches = x_patches * attention_scores
        attended_patches = self.activation(attended_patches)
        attended_patches = attended_patches.view(x.shape[0], -1, x.shape[-1])
        return torch.cat((cls_token, attended_patches), dim=1), attention_scores

class MultiScaleConvolutionalPatchSpatialAttention(torch.nn.Module):
    def __init__(self, num_patches, channels, patch_size=14, dropout_rate=0.1):
        super().__init__()
        self.conv_query1 = torch.nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=3, padding=1)
        self.conv_query2 = torch.nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=5, padding=2)
        self.conv_query3 = torch.nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=7, padding=3)
        self.batch_norm = torch.nn.BatchNorm2d(1)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = torch.nn.LeakyReLU(negative_slope=0.01)
        self.patch_size = patch_size

    def forward(self, x):
        cls_token = x[:, 0:1, :]
        x_patches = x[:, 1:, :]
        x_patches = x_patches.view(x.shape[0], -1, self.patch_size, self.patch_size)
        query1 = self.conv_query1(x_patches)
        query2 = self.conv_query2(x_patches)
        query3 = self.conv_query3(x_patches)
        query = query1 + query2 + query3
        query = self.batch_norm(query)
        query = query.view(x.shape[0], -1).unsqueeze(-1)
        attention_scores = torch.softmax(query, dim=1)
        attention_scores = self.dropout(attention_scores)
        attention_scores = attention_scores.view(x.shape[0], 1, self.patch_size, self.patch_size)
        attended_patches = x_patches * attention_scores
        attended_patches = self.activation(attended_patches)
        attended_patches = attended_patches.view(x.shape[0], -1, x.shape[-1])
        return torch.cat((cls_token, attended_patches), dim=1), attention_scores


class MultiHeadConvolutionalPatchSpatialAttention(torch.nn.Module):
    def __init__(self, num_patches, channels, patch_size=14, num_heads=3, dropout_rate=0.1):
        super().__init__()
        self.heads = torch.nn.ModuleList([
            MultiScaleConvolutionalPatchSpatialAttention(num_patches, channels, patch_size, dropout_rate)
            for _ in range(num_heads)
        ])
        self.combine_heads = torch.nn.Linear(num_heads, 1)

    def forward(self, x):
        head_outputs = []
        attention_scores = []
        for head in self.heads:
            attended, scores = head(x)
            head_outputs.append(attended)
            attention_scores.append(scores)
        stacked_outputs = torch.stack(head_outputs, dim=1)
        stacked_scores = torch.stack(attention_scores, dim=1)
        combined_output = torch.sum(stacked_outputs, dim=1)
        combined_scores = torch.mean(stacked_scores, dim=1)
        return combined_output, combined_scores


class PatchChannelAttention(torch.nn.Module):
    def __init__(self, num_features):
        super(PatchChannelAttention, self).__init__()
        self.fc1 = torch.nn.Linear(num_features, num_features // 8)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(num_features // 8, num_features)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        patch_embeddings = x[:, 1:, :]
        avg_pooled = torch.mean(patch_embeddings, dim=1, keepdim=True)
        max_pooled = torch.max(patch_embeddings, dim=1, keepdim=True)[0]
        scale = self.sigmoid(self.fc2(self.relu(self.fc1(avg_pooled))) + self.fc2(self.relu(self.fc1(max_pooled))))
        attended_patches = patch_embeddings * scale.expand_as(patch_embeddings)
        return torch.cat((x[:, :1, :], attended_patches), dim=1), scale

class PoolingToConcept(torch.nn.Module):
    def __init__(self, in_feature, emb_size):
        super(PoolingToConcept, self).__init__()
        self.fc = torch.nn.Linear(in_feature, 2*emb_size)
    def forward(self, x):
        x_pooled = torch.mean(x, dim=1)
        x_transformed = self.fc(x_pooled)
        return x_transformed


class AttentionPoolingToConcept(torch.nn.Module):
    def __init__(self, in_feature, emb_size):
        super(AttentionPoolingToConcept, self).__init__()
        self.attention_weights = torch.nn.Linear(in_feature, 1)
        self.fc = torch.nn.Linear(in_feature, 2*emb_size)
    def forward(self, x):
        weights = torch.softmax(self.attention_weights(x), dim=1)
        x_weighted = torch.sum(weights * x, dim=1)
        x_transformed = self.fc(x_weighted)
        return x_transformed

class AttentionConceptGenerator(torch.nn.Module):
    def __init__(self, emb_size):
        super(AttentionConceptGenerator, self).__init__()
        self.spatial_attention = MultiHeadConvolutionalPatchSpatialAttention(num_patches=196, channels=768, patch_size=14, num_heads=3, dropout_rate=0.01)
        self.channel_attention = PatchChannelAttention(num_features=768)
        self.attention_pooling = AttentionPoolingToConcept(in_feature=768, emb_size=emb_size)
        self.leakyrelu = torch.nn.LeakyReLU()

    def forward(self, x):
        x, spatial_att_score = self.spatial_attention(x)
        x, channel_att_score = self.channel_attention(x)
        x = self.attention_pooling(x)
        x = self.leakyrelu(x)
        return x, spatial_att_score, channel_att_score

class Attention_ConceptEmbeddingModel_ROI(ConceptBottleneckModel):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        emb_size=16,
        training_intervention_prob=0.25,
        embeding_activation="leakyrelu",
        shared_prob_gen=True,
        concept_loss_weight=1,
        map_loss_weight=1,
        task_loss_weight=1,
        c2y_model=None,
        c2y_layers=None,
        y_activation="sigmoid",
        c_extractor_arch=utils.wrap_pretrained_model(None, architecture=None),
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
        architecture=None,
        reg_concept=None,
    ):

        pl.LightningModule.__init__(self)
        self.n_concepts = n_concepts
        self.intervention_policy = intervention_policy
        self.pre_concept_model = c_extractor_arch(output_dim=None, architecture=architecture)
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
            generator = AttentionConceptGenerator(emb_size=emb_size)
            self.concept_context_generators.append(generator)
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
        if reg_concept == "huber":
            self.loss_concept = torch.nn.SmoothL1Loss(beta=1.0)
        else:
            self.loss_concept = torch.nn.BCELoss(weight=weight_loss)
        self.loss_task = (
            torch.nn.CrossEntropyLoss(weight=task_class_weights)
            if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(
                weight=task_class_weights
            )
        )
        self.loss_map = cosine_similarity_attention_loss
        self.concept_loss_weight = concept_loss_weight
        self.map_loss_weight = map_loss_weight
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
            spatial_att_scores = []
            channel_att_scores = []
            c_sem = []
            for i, context_gen in enumerate(self.concept_context_generators):
                if self.shared_prob_gen:
                    prob_gen = self.concept_prob_generators[0]
                else:
                    prob_gen = self.concept_prob_generators[i]
                context, spatial_att_score, channel_att_score = context_gen(pre_c)
                prob = prob_gen(context)
                contexts.append(torch.unsqueeze(context, dim=1))
                spatial_att_scores.append(torch.unsqueeze(spatial_att_score, dim=1))
                channel_att_scores.append(torch.unsqueeze(channel_att_score, dim=1))
                c_sem.append(self.sig(prob))
            c_sem = torch.cat(c_sem, axis=-1)
            contexts = torch.cat(contexts, axis=1)
            squeezed_spatial_att_scores = [tensor.squeeze() for tensor in spatial_att_scores]
            spatial_att_scores = torch.stack(squeezed_spatial_att_scores, dim=1)
            squeezed_channel_att_scores = [tensor.squeeze(1).squeeze(1) for tensor in channel_att_scores]
            channel_att_scores = torch.stack(squeezed_channel_att_scores, dim=2)
            latent = contexts, c_sem, spatial_att_scores, channel_att_scores
        else:
            contexts, c_sem, spatial_att_scores, channel_att_scores = latent
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
        inact_c_prob = 1 - torch.unsqueeze(probs, dim=-1)
        inact_c_prob = torch.clamp(inact_c_prob, min=0)
        c_pred = (
                contexts[:, :, :self.emb_size] * torch.unsqueeze(probs, dim=-1) +
                contexts[:, :, self.emb_size:] * inact_c_prob
        )
        c_pred = c_pred.view((-1, self.emb_size * self.n_concepts))
        y = self.c2y_model(c_pred)
        if self.output_latent:
            return c_sem, c_pred, y, latent
        return c_sem, c_pred, y, spatial_att_scores, channel_att_scores
