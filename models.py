"""CMUPF (MRI + PET) and the single-modality CNN baseline."""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config



# --------------------------------------------------------------- encoder
class BayesianConv3DBlock(nn.Module):
    def __init__(self, cin, cout, p=0.3):
        super().__init__()
        self.conv = nn.Conv3d(cin, cout, 3, padding=1)
        self.bn = nn.BatchNorm3d(cout)
        self.dropout = nn.Dropout3d(p)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.dropout(self.relu(self.bn(self.conv(x))))


class BayesianEncoder(nn.Module):
    """4-stage 3D CNN with dropout kept active at test time (MC dropout)."""

    def __init__(self, feature_dim=256, p=0.3):
        super().__init__()
        self.conv1, self.pool1 = BayesianConv3DBlock(1, 32, p), nn.MaxPool3d(2)
        self.conv2, self.pool2 = BayesianConv3DBlock(32, 64, p), nn.MaxPool3d(2)
        self.conv3, self.pool3 = BayesianConv3DBlock(64, 128, p), nn.MaxPool3d(2)
        self.conv4, self.pool4 = BayesianConv3DBlock(128, 256, p), nn.MaxPool3d(2)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc_features = nn.Linear(256, feature_dim)
        self.fc_uncertainty = nn.Linear(256, feature_dim)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.pool4(self.conv4(x))
        x = self.gap(x).flatten(1)
        return self.fc_features(x), F.softplus(self.fc_uncertainty(x))

    def mc_forward(self, x, num_samples=10):
        fs, as_ = [], []
        for _ in range(num_samples):
            f, a = self.forward(x)
            fs.append(f)
            as_.append(a)
        fs, as_ = torch.stack(fs), torch.stack(as_)
        epi = fs.std(0) if fs.shape[0] > 1 else torch.zeros_like(fs[0])
        return fs.mean(0), epi, as_.mean(0)


# ------------------------------------------------------------ components
class SharedUncertaintySpace(nn.Module):
    def __init__(self, feature_dim=256, shared_dim=64):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(feature_dim * 2, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, shared_dim))

    def forward(self, features, uncertainty):
        return self.projection(torch.cat([features, uncertainty], dim=1))


class CrossModalAttention(nn.Module):
    """Uncertainty-guided attention from modality i onto modality j."""

    def __init__(self, shared_dim=64):
        super().__init__()
        self.query = nn.Linear(shared_dim, shared_dim)
        self.key = nn.Linear(shared_dim, shared_dim)
        self.value = nn.Linear(shared_dim, shared_dim)
        self.scale = shared_dim ** 0.5

    def forward(self, u_i, u_j, unc_i, unc_j):
        Q, K, V = self.query(u_i), self.key(u_j), self.value(u_j)

        if Config.ATTENTION_MODE == "off":
            # ablation: the attention branch contributes nothing
            z = torch.zeros_like(V)
            return z, torch.zeros(V.shape[0], 1, device=V.device)

        # trust j more when i is uncertain and j is not
        w = (0.0 if Config.ATTENTION_MODE == "no_guidance" else
             torch.log1p(unc_i.mean(1, keepdim=True))
             - torch.log1p(unc_j.mean(1, keepdim=True)))

        if Config.ATTENTION_MODE == "original":
            # (batch, batch): each sample attends over the others in its batch
            scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale + w
            attn = F.softmax(scores, dim=-1)
            return torch.matmul(attn, V), attn

        scores = (Q * K).sum(1, keepdim=True) / self.scale + w
        attn = torch.sigmoid(scores)
        return attn * V, attn


class UncertaintyTransferModule(nn.Module):
    """Learns f_src -> f_dst for each ordered pair; also serves imputation."""

    def __init__(self, feature_dim=256, modalities=("mri", "pet")):
        super().__init__()
        self.modalities = list(modalities)
        self.nets = nn.ModuleDict({
            f"{a}_to_{b}": nn.Sequential(
                nn.Linear(feature_dim, feature_dim), nn.ReLU(),
                nn.Linear(feature_dim, feature_dim))
            for a in self.modalities for b in self.modalities if a != b})

    def impute(self, src, dst, feat_src):
        return self.nets[f"{src}_to_{dst}"](feat_src)

    def forward(self, features):
        total, n = 0.0, 0
        for i, a in enumerate(self.modalities):
            for j, b in enumerate(self.modalities):
                if i == j:
                    continue
                total = total + F.mse_loss(self.impute(a, b, features[i]),
                                           features[j].detach())
                n += 1
        return total / max(n, 1)


# -------------------------------------------------------------- the model
class CMUPF(nn.Module):
    MODALITIES = ("mri", "pet")

    def __init__(self, feature_dim=None, shared_dim=None, num_classes=None, dropout=None):
        super().__init__()
        feature_dim = feature_dim or Config.FEATURE_DIM
        shared_dim = shared_dim or Config.SHARED_DIM
        num_classes = num_classes or Config.NUM_CLASSES
        dropout = Config.DROPOUT if dropout is None else dropout

        self.mri_encoder = BayesianEncoder(feature_dim, dropout)
        self.pet_encoder = BayesianEncoder(feature_dim, dropout)
        self.shared_space = SharedUncertaintySpace(feature_dim, shared_dim)
        self.cross_attention = CrossModalAttention(shared_dim)
        self.transfer_module = UncertaintyTransferModule(feature_dim, self.MODALITIES)

        n = len(self.MODALITIES)
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim * n + shared_dim * n, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3))
        self.head_mu = nn.Linear(256, num_classes)
        self.head_logvar = nn.Linear(256, num_classes)

        self.mri_classifier = nn.Linear(feature_dim, num_classes)
        self.pet_classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, mri, pet, missing_modality=None, num_mc_samples=None):
        S = num_mc_samples or (Config.MC_SAMPLES_TRAIN if self.training
                               else Config.MC_SAMPLES_EVAL)

        f_mri, e_mri, a_mri = self.mri_encoder.mc_forward(mri, S)
        f_pet, e_pet, a_pet = self.pet_encoder.mc_forward(pet, S)
        u_mri, u_pet = e_mri + a_mri, e_pet + a_pet

        if self.training and missing_modality is None and Config.MODALITY_DROPOUT > 0:
            r = random.random()
            if r < Config.MODALITY_DROPOUT / 2:
                missing_modality = "mri"
            elif r < Config.MODALITY_DROPOUT:
                missing_modality = "pet"

        if missing_modality == "mri":
            f_mri = (self.transfer_module.impute("pet", "mri", f_pet)
                     if Config.IMPUTE_FROM_TRANSFER else torch.zeros_like(f_mri))
            u_mri = u_pet + Config.IMPUTE_UNC_PENALTY
        elif missing_modality == "pet":
            f_pet = (self.transfer_module.impute("mri", "pet", f_mri)
                     if Config.IMPUTE_FROM_TRANSFER else torch.zeros_like(f_pet))
            u_pet = u_mri + Config.IMPUTE_UNC_PENALTY

        s_mri = self.shared_space(f_mri, u_mri)
        s_pet = self.shared_space(f_pet, u_pet)
        att_mri, w_mri = self.cross_attention(s_mri, s_pet, u_mri, u_pet)
        att_pet, w_pet = self.cross_attention(s_pet, s_mri, u_pet, u_mri)

        h = self.trunk(torch.cat([f_mri, f_pet, att_mri, att_pet], dim=1))
        return {
            "logits": self.head_mu(h),
            "logvar": self.head_logvar(h).clamp(-6, 4),
            "modality_logits": {"mri": self.mri_classifier(f_mri),
                                "pet": self.pet_classifier(f_pet)},
            "shared": {"mri": s_mri, "pet": s_pet},
            "attn": {"mri": w_mri.detach(), "pet": w_pet.detach()},
            "uncertainties": {
                "mri": {"epistemic": e_mri, "aleatoric": a_mri, "total": u_mri},
                "pet": {"epistemic": e_pet, "aleatoric": a_pet, "total": u_pet}},
            "transfer_loss": self.transfer_module([f_mri, f_pet]),
            "missing": missing_modality,
        }


class CMUPFLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, out, y):
        if Config.HETEROSCEDASTIC:
            std = torch.exp(0.5 * out["logvar"])
            task = torch.stack([
                self.ce(out["logits"] + std * torch.randn_like(std), y)
                for _ in range(Config.HETERO_SAMPLES)]).mean()
        else:
            task = self.ce(out["logits"], y)

        aux = (self.ce(out["modality_logits"]["mri"], y) +
               self.ce(out["modality_logits"]["pet"], y)) / 2

        if Config.CONSISTENCY_SPACE == "raw":
            cons = F.mse_loss(out["uncertainties"]["mri"]["total"],
                              out["uncertainties"]["pet"]["total"])
        else:
            cons = F.mse_loss(out["shared"]["mri"], out["shared"]["pet"].detach())

        total = (Config.TASK_WEIGHT * task + Config.AUX_WEIGHT * aux +
                 Config.CONSISTENCY_WEIGHT * cons +
                 Config.TRANSFER_WEIGHT * out["transfer_loss"])
        return {"total_loss": total, "task_loss": task, "aux_loss": aux,
                "consistency_loss": cons, "transfer_loss": out["transfer_loss"]}


# ------------------------------------------------------------- baselines
class SingleModalityCNN(nn.Module):
    """Same encoder backbone, one modality, plain classifier.

    Capacity-matched to a single CMUPF branch so the comparison is fair.
    """

    def __init__(self, feature_dim=None, num_classes=None, dropout=None):
        super().__init__()
        feature_dim = feature_dim or Config.FEATURE_DIM
        num_classes = num_classes or Config.NUM_CLASSES
        dropout = Config.DROPOUT if dropout is None else dropout
        self.encoder = BayesianEncoder(feature_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes))

    def forward(self, x, num_mc_samples=None):
        S = num_mc_samples or (Config.MC_SAMPLES_TRAIN if self.training
                               else Config.MC_SAMPLES_EVAL)
        f, epi, ale = self.encoder.mc_forward(x, S)
        return {"logits": self.classifier(f),
                "uncertainties": {"total": epi + ale}}


class SingleModalityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, out, y):
        loss = self.ce(out["logits"], y)
        return {"total_loss": loss, "task_loss": loss}


# ------------------------------------------------- tabular (derived features)
class TabularEncoder(nn.Module):
    """MC-dropout MLP over a derived-feature vector.

    Same contract as BayesianEncoder — returns (features, aleatoric) and
    exposes mc_forward — so every downstream CMUPF component is unchanged.
    Replacing the 3D CNN with this is the only difference between CMUPF on
    voxels and CMUPF on measured anatomy.
    """

    def __init__(self, in_dim, feature_dim=64, hidden=128, p=0.3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(p))
        self.fc_features = nn.Linear(hidden, feature_dim)
        self.fc_uncertainty = nn.Linear(hidden, feature_dim)

    def forward(self, x):
        h = self.body(x)
        return self.fc_features(h), F.softplus(self.fc_uncertainty(h))

    def mc_forward(self, x, num_samples=10):
        fs, as_ = [], []
        for _ in range(num_samples):
            f, a = self.forward(x)
            fs.append(f)
            as_.append(a)
        fs, as_ = torch.stack(fs), torch.stack(as_)
        epi = fs.std(0) if fs.shape[0] > 1 else torch.zeros_like(fs[0])
        return fs.mean(0), epi, as_.mean(0)


class CMUPFTabular(nn.Module):
    """CMUPF over derived features: structural volumes + PET summaries.

    Identical to CMUPF except the encoders. Uncertainty (MC-dropout epistemic
    + predicted aleatoric), the shared uncertainty space, uncertainty-guided
    cross-modal attention, the transfer module, modality dropout with
    transfer-based imputation and the heteroscedastic head all behave exactly
    as in the voxel model — so a difference in results is attributable to the
    input representation, not to the framework.
    """

    MODALITIES = ("mri", "pet")

    def __init__(self, mri_dim, pet_dim, feature_dim=None, shared_dim=None,
                 hidden=None, num_classes=None, dropout=None):
        super().__init__()
        feature_dim = feature_dim or getattr(Config, "TAB_FEATURE_DIM", 64)
        shared_dim = shared_dim or getattr(Config, "TAB_SHARED_DIM", 32)
        hidden = hidden or getattr(Config, "TAB_HIDDEN", 128)
        num_classes = num_classes or getattr(Config, "NUM_CLASSES", 2)
        dropout = getattr(Config, "DROPOUT", 0.3) if dropout is None else dropout

        self.mri_encoder = TabularEncoder(mri_dim, feature_dim, hidden, dropout)
        self.pet_encoder = TabularEncoder(pet_dim, feature_dim, hidden, dropout)
        self.shared_space = SharedUncertaintySpace(feature_dim, shared_dim)
        self.cross_attention = CrossModalAttention(shared_dim)
        self.transfer_module = UncertaintyTransferModule(feature_dim, self.MODALITIES)

        n = len(self.MODALITIES)
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim * n + shared_dim * n, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3))
        self.head_mu = nn.Linear(64, num_classes)
        self.head_logvar = nn.Linear(64, num_classes)
        self.mri_classifier = nn.Linear(feature_dim, num_classes)
        self.pet_classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, mri, pet, missing_modality=None, num_mc_samples=None):
        S = num_mc_samples or (Config.MC_SAMPLES_TRAIN if self.training
                               else Config.MC_SAMPLES_EVAL)

        f_mri, e_mri, a_mri = self.mri_encoder.mc_forward(mri, S)
        f_pet, e_pet, a_pet = self.pet_encoder.mc_forward(pet, S)
        u_mri, u_pet = e_mri + a_mri, e_pet + a_pet

        if self.training and missing_modality is None and Config.MODALITY_DROPOUT > 0:
            r = random.random()
            if r < Config.MODALITY_DROPOUT / 2:
                missing_modality = "mri"
            elif r < Config.MODALITY_DROPOUT:
                missing_modality = "pet"

        if missing_modality == "mri":
            f_mri = (self.transfer_module.impute("pet", "mri", f_pet)
                     if Config.IMPUTE_FROM_TRANSFER else torch.zeros_like(f_mri))
            u_mri = u_pet + Config.IMPUTE_UNC_PENALTY
        elif missing_modality == "pet":
            f_pet = (self.transfer_module.impute("mri", "pet", f_mri)
                     if Config.IMPUTE_FROM_TRANSFER else torch.zeros_like(f_pet))
            u_pet = u_mri + Config.IMPUTE_UNC_PENALTY

        s_mri = self.shared_space(f_mri, u_mri)
        s_pet = self.shared_space(f_pet, u_pet)
        att_mri, w_mri = self.cross_attention(s_mri, s_pet, u_mri, u_pet)
        att_pet, w_pet = self.cross_attention(s_pet, s_mri, u_pet, u_mri)

        h = self.trunk(torch.cat([f_mri, f_pet, att_mri, att_pet], dim=1))
        return {
            "logits": self.head_mu(h),
            "logvar": self.head_logvar(h).clamp(-6, 4),
            "modality_logits": {"mri": self.mri_classifier(f_mri),
                                "pet": self.pet_classifier(f_pet)},
            "shared": {"mri": s_mri, "pet": s_pet},
            "attn": {"mri": w_mri.detach(), "pet": w_pet.detach()},
            "uncertainties": {
                "mri": {"epistemic": e_mri, "aleatoric": a_mri, "total": u_mri},
                "pet": {"epistemic": e_pet, "aleatoric": a_pet, "total": u_pet}},
            "transfer_loss": self.transfer_module([f_mri, f_pet]),
            "missing": missing_modality,
        }
