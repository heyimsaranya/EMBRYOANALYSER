import io
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

try:
    import timm
except ImportError:
    timm = None


st.set_page_config(page_title="Embryo Analyzer", page_icon="", layout="wide", initial_sidebar_state="expanded")

STAGE_DETAILS = {
    "tPB2": "Second polar body extrusion",
    "tPNa": "Pronuclei appear",
    "tPNf": "Pronuclei fading",
    "t2": "2-cell stage",
    "t3": "3-cell stage",
    "t4": "4-cell stage",
    "t5": "5-cell stage",
    "t6": "6-cell stage",
    "t7": "7-cell stage",
    "t8": "8-cell stage",
    "t9+": "9+ cell stage",
    "tM": "Morula",
    "tSB": "Start of blastulation",
    "tB": "Blastocyst",
    "tEB": "Expanded blastocyst",
}
STAGE_CLASSES = list(STAGE_DETAILS.keys())
VIABILITY_CLASSES = ["Non-viable", "Viable"]


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0b1220;
            --panel: #111a2e;
            --panel-soft: #17233d;
            --border: #2b3d63;
            --text: #edf4ff;
            --muted: #97abc9;
            --accent: #67b7ff;
            --danger: #ff7f88;
            --warn: #ffc76b;
            --success: #75d69a;
        }
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(44, 95, 165, 0.18), transparent 28%),
                linear-gradient(180deg, #08101d 0%, #0b1220 45%, #0d1728 100%);
            color: var(--text);
        }
        .main .block-container {
            max-width: 1380px;
            padding-top: 1.2rem;
            padding-bottom: 2.4rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a1221 0%, #0d1728 100%);
            border-right: 1px solid var(--border);
        }
        h1, h2, h3, h4, h5, h6, p, li, label, div {
            color: var(--text);
        }
        .panel, .hero-card {
            background: rgba(17, 26, 46, 0.92);
            border: 1px solid rgba(103, 183, 255, 0.16);
            border-radius: 18px;
            padding: 1rem 1.15rem;
            box-shadow: 0 14px 32px rgba(0, 0, 0, 0.18);
        }
        .hero-card {
            padding: 1.2rem 1.3rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            background: linear-gradient(180deg, rgba(23, 35, 61, 0.98), rgba(14, 24, 42, 0.98));
            border: 1px solid rgba(103, 183, 255, 0.16);
            border-radius: 16px;
            padding: 0.95rem 1rem;
            min-height: 110px;
        }
        .metric-label {
            color: var(--muted);
            font-size: 0.88rem;
            margin-bottom: 0.35rem;
        }
        .metric-value {
            font-size: 1.7rem;
            font-weight: 700;
            line-height: 1.1;
        }
        .metric-sub {
            color: var(--muted);
            font-size: 0.85rem;
            margin-top: 0.35rem;
        }
        .badge {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            font-size: 0.84rem;
            font-weight: 600;
            margin-right: 0.4rem;
            margin-bottom: 0.4rem;
        }
        .badge-info { background: rgba(103, 183, 255, 0.14); color: #cfe7ff; border: 1px solid rgba(103, 183, 255, 0.22); }
        .badge-success { background: rgba(117, 214, 154, 0.15); color: #d8f8e1; border: 1px solid rgba(117, 214, 154, 0.22); }
        .badge-warn { background: rgba(255, 199, 107, 0.16); color: #ffecc6; border: 1px solid rgba(255, 199, 107, 0.26); }
        .badge-danger { background: rgba(255, 127, 136, 0.14); color: #ffd6da; border: 1px solid rgba(255, 127, 136, 0.24); }
        .small-note { color: var(--muted); font-size: 0.88rem; }
        .stTabs [data-baseweb="tab-list"] button {
            background: rgba(17, 26, 46, 0.92);
            border: 1px solid rgba(103, 183, 255, 0.16);
            border-radius: 12px;
            margin-right: 0.5rem;
        }
        .stTabs [aria-selected="true"] { background: rgba(103, 183, 255, 0.16) !important; }
        div[data-testid="stMetric"] {
            background: rgba(17, 26, 46, 0.92);
            border: 1px solid rgba(103, 183, 255, 0.16);
            padding: 0.8rem;
            border-radius: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, subtext: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_badge(label: str, tone: str = "info") -> str:
    return f'<span class="badge badge-{tone}">{label}</span>'


@dataclass
class ClinicalInputs:
    maternal_age: Optional[int]
    previous_ivf_attempts: Optional[int]
    amh: Optional[float]
    fsh: Optional[float]
    fertilization_method: str
    embryo_culture_day: Optional[int]
    abnormalities: str
    clinician_notes: str


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def cv_crop_embryo(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return frame
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    x1, y1 = max(x - 20, 0), max(y - 20, 0)
    x2, y2 = min(x + w + 20, frame.shape[1]), min(y + h + 20, frame.shape[0])
    cropped = frame[y1:y2, x1:x2]
    if cropped.shape[0] < 10 or cropped.shape[1] < 10:
        return frame
    return cropped


def cv_luminosity_correction(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    l_chan = cv2.equalizeHist(l_chan)
    return cv2.cvtColor(cv2.merge([l_chan, a_chan, b_chan]), cv2.COLOR_LAB2RGB)


def preprocess_stage_frame(image: Image.Image) -> np.ndarray:
    frame = pil_to_rgb_array(image)
    frame = cv_crop_embryo(frame)
    frame = cv_luminosity_correction(frame)
    return frame


def sort_frame_files(files: Sequence) -> List:
    def frame_key(item) -> Tuple[int, str]:
        name = item.name
        match = re.search(r"RUN(\d+)", name, re.IGNORECASE)
        if match:
            return int(match.group(1)), name
        numeric = re.findall(r"\d+", name)
        return (int(numeric[-1]) if numeric else 10**9, name)

    return sorted(files, key=frame_key)


class CNNEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        base = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
        self.feature = nn.Sequential(*list(base.children())[:-1])
        self.out_dim = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        return features.view(features.size(0), -1)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8) -> None:
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [
            tensor.reshape(batch, tokens, self.heads, dim // self.heads).transpose(1, 2)
            for tensor in qkv
        ]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TemporalViT(nn.Module):
    def __init__(self, num_classes: int = 15, dim: int = 2048, depth: int = 6, heads: int = 8) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1000, dim))
        self.blocks = nn.Sequential(*[TransformerBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, frames, dim = x.shape
        cls = self.cls_token.expand(batch, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed[:, : frames + 1, :]
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])


class CNNViTHybrid(nn.Module):
    def __init__(self, num_classes: int = 15) -> None:
        super().__init__()
        self.cnn = CNNEncoder()
        self.vit = TemporalViT(num_classes=num_classes, dim=2048)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(1)
        batch, frames, channels, height, width = x.shape
        x = x.view(batch * frames, channels, height, width)
        feats = self.cnn(x).view(batch, frames, -1)
        return self.vit(feats)


class HybridEmbryoClassifier(nn.Module):
    def __init__(self, model_name: str = "tf_efficientnetv2_s", num_classes: int = 2, dropout_rate: float = 0.5):
        super().__init__()
        self.model_name = model_name
        self.num_features = 0
        self.backbone, self.num_features = self._load_backbone(model_name)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout_rate * 0.8),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _load_backbone(self, model_name: str):
        if timm is None:
            raise ImportError("timm is required for the viability model.")
        backbone = timm.create_model(model_name, pretrained=True)
        if hasattr(backbone, "classifier"):
            num_features = backbone.classifier.in_features if isinstance(backbone.classifier, nn.Linear) else getattr(backbone, "num_features", 1280)
            backbone.classifier = nn.Identity()
            return backbone, num_features
        if hasattr(backbone, "fc"):
            num_features = backbone.fc.in_features if isinstance(backbone.fc, nn.Linear) else getattr(backbone, "num_features", 2048)
            backbone.fc = nn.Identity()
            return backbone, num_features
        if hasattr(backbone, "head"):
            num_features = backbone.head.in_features if isinstance(backbone.head, nn.Linear) else getattr(backbone, "num_features", 1280)
            backbone.head = nn.Identity()
            return backbone, num_features
        return backbone, getattr(backbone, "num_features", 1280)

    def _init_weights(self) -> None:
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.dim() == 4:
            features = self.gap(features).flatten(1)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._extract_features(x))


def infer_backbone_name(state_dict: Dict[str, torch.Tensor]) -> str:
    classifier_weight = state_dict.get("classifier.1.weight")
    if classifier_weight is None:
        return "tf_efficientnetv2_s"
    in_features = classifier_weight.shape[1]
    if in_features == 2048:
        return "resnet50"
    if in_features == 1536:
        return "efficientnet_b3"
    return "tf_efficientnetv2_s"


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def discover_model_candidates() -> Dict[str, str]:
    search_roots = [Path.cwd(), Path.home() / "Downloads"]
    candidates: Dict[str, str] = {}
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.glob("*.pth"):
            candidates[path.name] = str(path)
    return dict(sorted(candidates.items()))


@st.cache_resource(show_spinner=False)
def load_stage_model(model_path: str, device_str: str):
    device = torch.device(device_str)
    checkpoint = torch.load(model_path, map_location=device)
    model = CNNViTHybrid(num_classes=len(STAGE_CLASSES)).to(device)
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    model.eval()
    return model, checkpoint


@st.cache_resource(show_spinner=False)
def load_viability_model(model_path: str, device_str: str):
    device = torch.device(device_str)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    backbone_name = infer_backbone_name(state_dict)
    model = HybridEmbryoClassifier(model_name=backbone_name, num_classes=2, dropout_rate=0.5).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, checkpoint, backbone_name


def viability_transform() -> T.Compose:
    return T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def stage_transform() -> T.Compose:
    return T.Compose(
        [
            T.ToPILImage(),
            T.Resize((256, 256)),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def tensor_from_image(image: Image.Image, task: str) -> torch.Tensor:
    if task == "stage":
        return stage_transform()(preprocess_stage_frame(image))
    return viability_transform()(image.convert("RGB"))


def softmax_confidence(logits: torch.Tensor) -> Tuple[int, float, np.ndarray]:
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return pred_idx, float(probs[pred_idx]), probs


def get_risk_level(label: str, confidence: float) -> str:
    if label == "Viable":
        if confidence >= 0.75:
            return "Low"
        if confidence >= 0.55:
            return "Moderate"
        return "High"
    if confidence >= 0.75:
        return "High"
    return "Moderate"


def uncertainty_flag(confidence: float, secondary_gap: float) -> str:
    if confidence < 0.55 or secondary_gap < 0.10:
        return "Flagged"
    return "Not flagged"


def context_insights(inputs: ClinicalInputs) -> List[str]:
    insights: List[str] = []
    if inputs.maternal_age is not None:
        if inputs.maternal_age >= 38:
            insights.append("Advanced maternal age can increase biological variability; review low-confidence outputs carefully.")
        elif inputs.maternal_age < 30:
            insights.append("Relatively favorable maternal age is noted, but predictions still require full embryology review.")
    if inputs.previous_ivf_attempts is not None and inputs.previous_ivf_attempts >= 2:
        insights.append("Multiple prior IVF attempts may justify closer review of uncertain cases and correlation with historical outcomes.")
    if inputs.amh is not None and inputs.amh < 1.0:
        insights.append("Low AMH may indicate reduced ovarian reserve; this is contextual only and does not override image-based output.")
    if inputs.fsh is not None and inputs.fsh > 10:
        insights.append("Elevated FSH is clinically relevant context and may support a more cautious interpretation.")
    if inputs.embryo_culture_day is not None:
        insights.append(f"Entered culture day: Day {inputs.embryo_culture_day}. Stage predictions should be checked against expected developmental timing.")
    if inputs.abnormalities.strip():
        insights.append("Known abnormalities were entered; direct visual review and explanation maps should be prioritized.")
    if not insights:
        insights.append("No contextual modifiers were entered. Predictions are shown as assistive outputs only.")
    return insights


def find_last_conv_layer(model: nn.Module) -> Optional[nn.Module]:
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv


def generate_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    target_layer: Optional[nn.Module] = None,
) -> np.ndarray:
    model.eval()
    gradients = []
    activations = []
    if target_layer is None:
        target_layer = find_last_conv_layer(model)
    if target_layer is None:
        raise RuntimeError("No convolution layer found for Grad-CAM.")

    def forward_hook(_, __, output):
        activations.append(output.detach())

    def backward_hook(_, grad_input, grad_output):
        del grad_input
        gradients.append(grad_output[0].detach())

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)
    try:
        logits = model(input_tensor)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        score = logits[:, target_class].sum()
        model.zero_grad(set_to_none=True)
        score.backward()
        grads = gradients[-1]
        acts = activations[-1]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    finally:
        handle_f.remove()
        handle_b.remove()


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    rgb = np.array(image.convert("RGB").resize((224, 224)))
    heatmap_uint8 = np.uint8(255 * heatmap)
    color_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(rgb, 1 - alpha, color_map, alpha, 0)
    return Image.fromarray(blended)


def predict_viability(model: nn.Module, image: Image.Image, device: torch.device):
    tensor = tensor_from_image(image, "viability").unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    pred_idx, confidence, probs = softmax_confidence(logits)
    sorted_probs = np.sort(probs)[::-1]
    gap = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else float(sorted_probs[0])
    label = VIABILITY_CLASSES[pred_idx]
    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probs,
        "risk_level": get_risk_level(label, confidence),
        "uncertainty": uncertainty_flag(confidence, gap),
        "tensor": tensor,
        "pred_idx": pred_idx,
        "gap": gap,
    }


def rolling_average(probabilities: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return probabilities.copy()
    smoothed = []
    for idx in range(len(probabilities)):
        start = max(0, idx - window + 1)
        smoothed.append(probabilities[start : idx + 1].mean(axis=0))
    return np.vstack(smoothed)


def enforce_monotonic_progression(pred_indices: List[int]) -> List[int]:
    if not pred_indices:
        return pred_indices
    corrected = [pred_indices[0]]
    for idx in pred_indices[1:]:
        corrected.append(max(corrected[-1], idx))
    return corrected


def predict_stage_sequence(
    model: nn.Module,
    images: Sequence[Image.Image],
    device: torch.device,
    smoothing_window: int = 3,
    enforce_progression: bool = True,
):
    frame_tensors = [tensor_from_image(image, "stage") for image in images]
    batch = torch.stack(frame_tensors).to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = F.softmax(logits, dim=1).cpu().numpy()
    probs = rolling_average(probs, window=smoothing_window)
    pred_indices = probs.argmax(axis=1).tolist()
    if enforce_progression:
        pred_indices = enforce_monotonic_progression(pred_indices)

    rows = []
    for idx, (pred_idx, prob_vector) in enumerate(zip(pred_indices, probs), start=1):
        confidence = float(prob_vector[pred_idx])
        second_best = float(np.partition(prob_vector, -2)[-2]) if len(prob_vector) > 1 else 0.0
        rows.append(
            {
                "frame_index": idx,
                "stage": STAGE_CLASSES[pred_idx],
                "stage_description": STAGE_DETAILS[STAGE_CLASSES[pred_idx]],
                "confidence": confidence,
                "uncertainty": uncertainty_flag(confidence, confidence - second_best),
            }
        )

    frame_df = pd.DataFrame(rows)
    overall_stage = frame_df.iloc[-1]["stage"] if not frame_df.empty else "N/A"
    overall_conf = float(frame_df.iloc[-1]["confidence"]) if not frame_df.empty else 0.0
    return {
        "frame_df": frame_df,
        "probabilities": probs,
        "overall_stage": overall_stage,
        "overall_description": STAGE_DETAILS.get(overall_stage, "N/A"),
        "overall_confidence": overall_conf,
        "frame_tensors": frame_tensors,
    }


def fig_template(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#edf4ff"),
        margin=dict(l=30, r=20, t=40, b=30),
    )
    return fig


def build_dashboard_figures(viability_results: List[Dict], stage_results: List[pd.DataFrame]):
    figures = {}
    if viability_results:
        viability_df = pd.DataFrame(
            [
                {
                    "label": item["label"],
                    "confidence": item["confidence"] * 100,
                    "risk_level": item["risk_level"],
                    "uncertainty": item["uncertainty"],
                }
                for item in viability_results
            ]
        )
        fig = px.histogram(
            viability_df,
            x="label",
            color="label",
            category_orders={"label": VIABILITY_CLASSES},
            color_discrete_map={"Viable": "#75d69a", "Non-viable": "#ff7f88"},
        )
        figures["viability_distribution"] = fig_template(fig)

    if stage_results:
        stage_df = pd.concat(stage_results, ignore_index=True)
        fig = px.histogram(
            stage_df,
            x="stage",
            color="stage",
            category_orders={"stage": STAGE_CLASSES},
            color_discrete_sequence=px.colors.sequential.Blues,
        )
        figures["stage_distribution"] = fig_template(fig)
        trend = px.line(
            stage_df,
            x="frame_index",
            y=stage_df["stage"].map({name: idx for idx, name in enumerate(STAGE_CLASSES)}),
            markers=True,
            labels={"y": "Stage Index", "frame_index": "Frame"},
        )
        trend.update_yaxes(tickmode="array", tickvals=list(range(len(STAGE_CLASSES))), ticktext=STAGE_CLASSES)
        figures["stage_progression"] = fig_template(trend)
    return figures


def stage_probability_chart(frame_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame_df["frame_index"],
            y=frame_df["confidence"] * 100,
            mode="lines+markers",
            line=dict(color="#67b7ff", width=3),
            marker=dict(size=8),
            name="Stage confidence",
        )
    )
    fig.update_layout(xaxis_title="Frame", yaxis_title="Confidence (%)", yaxis_range=[0, 100])
    return fig_template(fig)


def probability_bar(labels: List[str], values: np.ndarray, colors_map: Dict[str, str]) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=(values * 100).tolist(),
                marker_color=[colors_map.get(label, "#67b7ff") for label in labels],
            )
        ]
    )
    fig.update_layout(yaxis_title="Probability (%)", xaxis_title="")
    return fig_template(fig)


def image_bytes_for_report(image: Image.Image, max_size: Tuple[int, int] = (480, 480)) -> io.BytesIO:
    img = image.copy()
    img.thumbnail(max_size)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def build_case_report_pdf(
    case_title: str,
    image: Optional[Image.Image],
    viability_summary: Optional[Dict],
    stage_summary: Optional[Dict],
    inputs: ClinicalInputs,
) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ClinicalTitle", parent=styles["Heading1"], textColor=colors.HexColor("#0b1220")))
    styles.add(ParagraphStyle(name="ClinicalBody", parent=styles["BodyText"], leading=14))
    story = [Paragraph(f"Embryo Analyzer Case Summary: {case_title}", styles["ClinicalTitle"]), Spacer(1, 0.18 * inch)]
    story.append(
        Paragraph(
            "Assistive research summary for clinician review. This output is not a standalone diagnosis and should be interpreted with laboratory, morphologic, and patient-context information.",
            styles["ClinicalBody"],
        )
    )
    story.append(Spacer(1, 0.18 * inch))
    if image is not None:
        story.append(RLImage(image_bytes_for_report(image), width=2.4 * inch, height=2.4 * inch))
        story.append(Spacer(1, 0.15 * inch))

    rows = [["Field", "Value"]]
    if viability_summary:
        rows.extend(
            [
                ["Viability prediction", viability_summary["label"]],
                ["Viability confidence", f'{viability_summary["confidence"] * 100:.1f}%'],
                ["Risk level", viability_summary["risk_level"]],
                ["Uncertainty flag", viability_summary["uncertainty"]],
            ]
        )
    if stage_summary:
        rows.extend(
            [
                ["Most advanced detected stage", stage_summary["overall_stage"]],
                ["Stage confidence", f'{stage_summary["overall_confidence"] * 100:.1f}%'],
                ["Stage detail", stage_summary["overall_description"]],
            ]
        )
    rows.extend(
        [
            ["Maternal age", str(inputs.maternal_age or "Not provided")],
            ["Previous IVF attempts", str(inputs.previous_ivf_attempts or "Not provided")],
            ["AMH", str(inputs.amh or "Not provided")],
            ["FSH", str(inputs.fsh or "Not provided")],
            ["Fertilization method", inputs.fertilization_method],
            ["Embryo culture day", str(inputs.embryo_culture_day or "Not provided")],
            ["Known abnormalities", inputs.abnormalities or "None reported"],
            ["Notes", inputs.clinician_notes or "None"],
        ]
    )

    table = Table(rows, colWidths=[2.2 * inch, 4.6 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce7f5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0b1220")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c2d0e6")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fb")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )
    story.append(table)
    doc.build(story)
    return buffer.getvalue()


def render_model_info(viability_backbone: Optional[str]) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Model Info")
    st.markdown(
        """
        - `Viability model`: Hybrid CNN classifier from your notebook with a TIMM backbone and custom dense head.
        - `Stage model`: ResNet50 encoder plus temporal transformer (`CNNViTHybrid`) from your notebook.
        - `Inference note`: Stage frames are predicted individually here because the notebook trained effectively with `T=1`.
        - `Confidence`: Displayed as softmax probability in percent, not raw logits.
        - `Role`: Assistive decision-support for embryo review, not autonomous diagnosis.
        """
        + (f"\n- `Detected viability backbone`: `{viability_backbone}`" if viability_backbone else "")
    )
    st.markdown("</div>", unsafe_allow_html=True)


def parse_optional_int(value: str) -> Optional[int]:
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_optional_float(value: str) -> Optional[float]:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def main() -> None:
    inject_styles()
    device = get_device()
    candidates = discover_model_candidates()

    st.markdown(
        """
        <div class="hero-card">
            <h1 style="margin-bottom:0.35rem;">Embryo Analyzer</h1>
            <p style="margin:0;color:#c8d9f5;">
                Clinical decision-support interface for embryo viability review and developmental stage analysis.
                Outputs are assistive, confidence-aware, and designed for clinician oversight.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## Navigation")
        st.markdown("Use the tabs for viability review, stage analysis, and analytics.")
        st.markdown("## Model Paths")
        model_options = [""] + list(candidates.values())
        default_stage = next(
            (value for value in model_options if value.endswith("best_stage_model.pth")),
            next((value for value in model_options if value.endswith("best_model.pth")), ""),
        )
        default_viability = next((value for value in model_options if value.endswith("best_embryo_model.pth")), "")
        stage_model_path = st.selectbox(
            "Stage model (.pth)",
            model_options,
            index=model_options.index(default_stage) if default_stage in model_options else 0,
        )
        viability_model_path = st.selectbox(
            "Viability model (.pth)",
            model_options,
            index=model_options.index(default_viability) if default_viability in model_options else 0,
        )
        st.markdown("## Inference Settings")
        smoothing_window = st.slider("Stage smoothing window", min_value=1, max_value=7, value=3)
        enforce_progression = st.toggle("Enforce monotonic stage progression", value=True)
        st.caption("Helpful for timelapse review when developmental stages should not move backward.")

    viability_model = None
    stage_model = None
    viability_backbone = None
    load_errors = []

    if stage_model_path:
        try:
            stage_model, _ = load_stage_model(stage_model_path, str(device))
        except Exception as exc:
            load_errors.append(f"Stage model could not be loaded: {exc}")

    if viability_model_path:
        try:
            viability_model, _, viability_backbone = load_viability_model(viability_model_path, str(device))
        except Exception as exc:
            load_errors.append(f"Viability model could not be loaded: {exc}")

    for error in load_errors:
        st.warning(error)

    st.markdown("## Dashboard Overview")
    viability_uploads = st.session_state.get("viability_uploads", [])
    stage_uploads = st.session_state.get("stage_uploads", [])
    viability_results_store = st.session_state.get("viability_results", [])
    stage_results_store = st.session_state.get("stage_results", [])
    figures = build_dashboard_figures(viability_results_store, [item["frame_df"] for item in stage_results_store])

    stats_cols = st.columns(4)
    with stats_cols[0]:
        render_metric_card("Uploaded viability cases", str(len(viability_uploads)), "Single-frame image reviews")
    with stats_cols[1]:
        render_metric_card("Uploaded timelapse frames", str(len(stage_uploads)), "Frames in current sequence")
    with stats_cols[2]:
        avg_conf = (
            f'{np.mean([item["confidence"] for item in viability_results_store]) * 100:.1f}%'
            if viability_results_store
            else "N/A"
        )
        render_metric_card("Mean viability confidence", avg_conf, "Across current batch")
    with stats_cols[3]:
        flagged = sum(1 for item in viability_results_store if item["uncertainty"] == "Flagged")
        render_metric_card("Flagged uncertain cases", str(flagged), "Require closer review")

    chart_cols = st.columns(2)
    with chart_cols[0]:
        if "viability_distribution" in figures:
            st.plotly_chart(figures["viability_distribution"], use_container_width=True)
        else:
            st.info("Upload viability images to populate the dashboard.")
    with chart_cols[1]:
        if "stage_distribution" in figures:
            st.plotly_chart(figures["stage_distribution"], use_container_width=True)
        else:
            st.info("Upload timelapse frames to populate stage distribution.")

    tabs = st.tabs(["Viability", "Stage Classification", "Analytics"])

    with tabs[0]:
        left, right = st.columns([1.0, 1.1], gap="large")
        with left:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.subheader("Image Upload Panel")
            viability_files = st.file_uploader(
                "Upload one or more embryo frames for viability classification",
                type=["png", "jpg", "jpeg", "tif", "tiff"],
                accept_multiple_files=True,
                key="viability_uploader",
            )
            if viability_files:
                st.session_state["viability_uploads"] = viability_files
                preview_cols = st.columns(min(3, len(viability_files)))
                for idx, file in enumerate(viability_files[:3]):
                    with preview_cols[idx % len(preview_cols)]:
                        st.image(file, caption=file.name, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.subheader("Clinical Questionnaire")
            col1, col2 = st.columns(2)
            with col1:
                maternal_age_text = st.text_input("Maternal age", placeholder="Optional")
                previous_ivf_attempts_text = st.text_input("Previous IVF attempts", placeholder="Optional")
                amh_text = st.text_input("AMH", placeholder="Optional")
            with col2:
                fsh_text = st.text_input("FSH", placeholder="Optional")
                fertilization_method = st.selectbox("Fertilization method", ["IVF", "ICSI", "Not specified"])
                embryo_culture_day_text = st.text_input("Embryo culture day", placeholder="Optional")
            abnormalities = st.text_area("Known abnormalities", placeholder="Optional embryology notes")
            clinician_notes = st.text_area("Notes section", placeholder="Optional report notes")
            st.markdown("</div>", unsafe_allow_html=True)

        inputs = ClinicalInputs(
            maternal_age=parse_optional_int(maternal_age_text),
            previous_ivf_attempts=parse_optional_int(previous_ivf_attempts_text),
            amh=parse_optional_float(amh_text),
            fsh=parse_optional_float(fsh_text),
            fertilization_method=fertilization_method,
            embryo_culture_day=parse_optional_int(embryo_culture_day_text),
            abnormalities=abnormalities,
            clinician_notes=clinician_notes,
        )

        if st.button("Run viability analysis", type="primary", disabled=not (viability_files and viability_model)):
            results = []
            for file in viability_files:
                image = Image.open(file).convert("RGB")
                result = predict_viability(viability_model, image, device)
                result["file_name"] = file.name
                result["image"] = image
                results.append(result)
            st.session_state["viability_results"] = results
            viability_results_store = results

        if viability_results_store:
            st.markdown("### Prediction Panel")
            for idx, result in enumerate(viability_results_store, start=1):
                c1, c2 = st.columns([0.9, 1.2], gap="large")
                with c1:
                    st.image(result["image"], caption=result["file_name"], use_container_width=True)
                    st.markdown(
                        render_badge(result["label"], "success" if result["label"] == "Viable" else "danger")
                        + render_badge(f'{result["confidence"] * 100:.1f}% confidence', "info")
                        + render_badge(
                            f'{result["risk_level"]} risk',
                            "warn" if result["risk_level"] == "Moderate" else ("success" if result["risk_level"] == "Low" else "danger"),
                        )
                        + render_badge(result["uncertainty"], "danger" if result["uncertainty"] == "Flagged" else "info"),
                        unsafe_allow_html=True,
                    )
                    st.progress(int(result["confidence"] * 100))

                with c2:
                    st.plotly_chart(
                        probability_bar(
                            VIABILITY_CLASSES,
                            result["probabilities"],
                            {"Viable": "#75d69a", "Non-viable": "#ff7f88"},
                        ),
                        use_container_width=True,
                    )
                    st.markdown("#### Clinical Decision Support")
                    st.write(
                        f"The model suggests **{result['label']}** with **{result['confidence'] * 100:.1f}%** confidence. "
                        f"Risk is flagged as **{result['risk_level']}**, and uncertainty is **{result['uncertainty']}**."
                    )
                    for insight in context_insights(inputs):
                        st.caption(f"- {insight}")

                    st.markdown("#### Explainability")
                    cam = generate_gradcam(
                        viability_model,
                        result["tensor"],
                        target_class=result["pred_idx"],
                        target_layer=find_last_conv_layer(viability_model.backbone),
                    )
                    overlay = overlay_heatmap(result["image"], cam)
                    mode = st.radio(
                        f"Heatmap view for case {idx}",
                        ["Original", "Grad-CAM Overlay"],
                        horizontal=True,
                        key=f"viability_cam_{idx}",
                    )
                    display_image = overlay if mode == "Grad-CAM Overlay" else result["image"].resize((224, 224))
                    st.image(display_image, use_container_width=True)

            selected = viability_results_store[0]
            report = build_case_report_pdf(
                case_title=selected["file_name"],
                image=selected["image"],
                viability_summary=selected,
                stage_summary=None,
                inputs=inputs,
            )
            st.download_button(
                "Download case summary report (PDF)",
                data=report,
                file_name="embryo_case_summary.pdf",
                mime="application/pdf",
            )

    with tabs[1]:
        left, right = st.columns([1.0, 1.0], gap="large")
        with left:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.subheader("Timelapse Upload Panel")
            stage_files = st.file_uploader(
                "Upload embryo timelapse frames",
                type=["png", "jpg", "jpeg", "tif", "tiff"],
                accept_multiple_files=True,
                key="stage_uploader",
            )
            if stage_files:
                ordered_files = sort_frame_files(stage_files)
                st.session_state["stage_uploads"] = ordered_files
                preview_cols = st.columns(min(4, len(ordered_files)))
                for idx, file in enumerate(ordered_files[:4]):
                    with preview_cols[idx % len(preview_cols)]:
                        st.image(file, caption=file.name, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.subheader("Stage Guide")
            guide_df = pd.DataFrame([{"Stage": stage, "Clinical Description": desc} for stage, desc in STAGE_DETAILS.items()])
            st.dataframe(guide_df, use_container_width=True, hide_index=True)
            st.markdown(
                """
                <p class="small-note">
                Developmental staging can support IVF review by helping embryologists identify progression patterns,
                highlight delayed or implausible transitions, and document sequence consistency across culture.
                </p>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Run stage analysis", type="primary", disabled=not (stage_files and stage_model)):
            ordered_files = sort_frame_files(stage_files)
            images = [Image.open(file).convert("RGB") for file in ordered_files]
            result = predict_stage_sequence(
                stage_model,
                images,
                device,
                smoothing_window=smoothing_window,
                enforce_progression=enforce_progression,
            )
            result["images"] = images
            result["file_names"] = [file.name for file in ordered_files]
            st.session_state["stage_results"] = [result]
            stage_results_store = [result]

        if stage_results_store:
            result = stage_results_store[0]
            frame_df = result["frame_df"]
            st.markdown("### Prediction Panel")
            metrics = st.columns(4)
            with metrics[0]:
                render_metric_card("Frames analyzed", str(len(frame_df)), "Frame-wise inference")
            with metrics[1]:
                render_metric_card("Latest stage", result["overall_stage"], result["overall_description"])
            with metrics[2]:
                render_metric_card("Latest confidence", f'{result["overall_confidence"] * 100:.1f}%', "Softmax confidence")
            with metrics[3]:
                flagged = int((frame_df["uncertainty"] == "Flagged").sum())
                render_metric_card("Uncertain frames", str(flagged), "Needs closer review")

            chart_cols = st.columns([1.1, 0.9], gap="large")
            with chart_cols[0]:
                st.plotly_chart(stage_probability_chart(frame_df), use_container_width=True)
            with chart_cols[1]:
                st.markdown(
                    render_badge(f"Latest stage: {result['overall_stage']}", "info")
                    + render_badge(result["overall_description"], "success")
                    + render_badge("Progression smoothing on" if enforce_progression else "Progression smoothing off", "warn"),
                    unsafe_allow_html=True,
                )

            st.dataframe(frame_df, use_container_width=True, hide_index=True)

            st.markdown("### Explainability")
            frame_index = st.slider("Select frame for Grad-CAM", min_value=1, max_value=len(result["images"]), value=1)
            selected_image = result["images"][frame_index - 1]
            selected_tensor = result["frame_tensors"][frame_index - 1].unsqueeze(0).to(device)
            selected_class = STAGE_CLASSES.index(frame_df.iloc[frame_index - 1]["stage"])
            cam = generate_gradcam(
                stage_model,
                selected_tensor,
                target_class=selected_class,
                target_layer=find_last_conv_layer(stage_model.cnn.feature),
            )
            overlay = overlay_heatmap(selected_image, cam)
            mode = st.radio("Stage heatmap view", ["Original", "Grad-CAM Overlay"], horizontal=True, key="stage_cam")
            st.image(overlay if mode == "Grad-CAM Overlay" else selected_image.resize((224, 224)), use_container_width=False)

            report = build_case_report_pdf(
                case_title="Timelapse sequence",
                image=selected_image,
                viability_summary=None,
                stage_summary=result,
                inputs=inputs,
            )
            st.download_button(
                "Download stage summary report (PDF)",
                data=report,
                file_name="embryo_stage_summary.pdf",
                mime="application/pdf",
            )

    with tabs[2]:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Trends and Analytics")
        if "stage_progression" in figures:
            st.plotly_chart(figures["stage_progression"], use_container_width=True)
        else:
            st.info("Run stage analysis to visualize progression.")

        if viability_results_store:
            analytics_df = pd.DataFrame(
                [
                    {
                        "file_name": result["file_name"],
                        "label": result["label"],
                        "confidence": result["confidence"] * 100,
                        "risk_level": result["risk_level"],
                        "uncertainty": result["uncertainty"],
                    }
                    for result in viability_results_store
                ]
            )
            st.dataframe(analytics_df, use_container_width=True, hide_index=True)
        else:
            st.info("Run viability analysis on one or more frames to populate batch analytics.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Clinical Research Context")
        st.markdown(
            """
            - `Why viability review matters`: A viability classifier can help prioritize embryos for further review, support consistency across readers, and surface uncertain cases that deserve deeper embryologist attention.
            - `Why stage detection matters`: Developmental stage timing is clinically meaningful because progression patterns can inform embryo selection strategy, culture monitoring, and annotation quality control.
            - `Assistive use`: The interface is designed to complement clinician judgment by combining predictions, confidence, uncertainty, clinical context, and explainability in one workspace.
            - `Important limitation`: Image-based outputs should not replace formal embryology assessment, patient-specific counseling, or laboratory standards.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    render_model_info(viability_backbone)


if __name__ == "__main__":
    main()
