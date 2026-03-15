import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from transformers import T5ForConditionalGeneration, T5Tokenizer
import timm
import io
import base64

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PlantGuard AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

:root {
    --green-dark:   #0d2818;
    --green-mid:    #1a4731;
    --green-leaf:   #2d6a4f;
    --green-bright: #52b788;
    --green-light:  #95d5b2;
    --cream:        #f8f4e8;
    --cream-dark:   #ede8d5;
    --amber:        #e9a825;
    --red-disease:  #c1440e;
    --text-dark:    #1a1a1a;
    --text-mid:     #3d3d3d;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--green-dark);
    color: var(--cream);
}

/* ── Main background ── */
.stApp {
    background: linear-gradient(145deg, #0a1f12 0%, #0d2818 40%, #0f3020 100%);
    background-attachment: fixed;
}

/* ── Hide streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero header ── */
.hero-banner {
    background: linear-gradient(135deg, #1a4731 0%, #0d2818 50%, #162d1f 100%);
    border: 1px solid #2d6a4f44;
    border-radius: 20px;
    padding: 48px 56px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    border-radius: 50%;
    background: radial-gradient(circle, #52b78822 0%, transparent 70%);
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -40px; left: -40px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, #2d6a4f33 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3.4rem;
    color: var(--cream);
    line-height: 1.1;
    margin: 0 0 12px 0;
    letter-spacing: -0.5px;
}
.hero-title span { color: var(--green-bright); font-style: italic; }
.hero-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.1rem;
    color: var(--green-light);
    font-weight: 300;
    margin: 0;
    letter-spacing: 0.3px;
}
.hero-badge {
    display: inline-block;
    background: #52b78822;
    border: 1px solid #52b78866;
    color: var(--green-bright);
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 16px;
    letter-spacing: 1px;
}

/* ── Cards ── */
.card {
    background: linear-gradient(145deg, #142b1e, #0f2216);
    border: 1px solid #2d6a4f33;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
    transition: border-color 0.3s;
}
.card:hover { border-color: #52b78866; }
.card-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: var(--green-light);
    margin: 0 0 16px 0;
}
.card-title span { font-size: 1.1rem; margin-right: 8px; }

/* ── Prediction result ── */
.prediction-healthy {
    background: linear-gradient(135deg, #1a4731 0%, #0f3020 100%);
    border: 2px solid #52b78888;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
}
.prediction-disease {
    background: linear-gradient(135deg, #2d1a0e 0%, #1a0f08 100%);
    border: 2px solid #c1440e88;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
}
.prediction-label {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    margin: 12px 0 6px 0;
    line-height: 1.2;
}
.prediction-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: var(--green-light);
    letter-spacing: 1px;
}

/* ── Confidence bar ── */
.conf-bar-wrap {
    background: #0a1f12;
    border-radius: 8px;
    height: 10px;
    margin: 12px 0;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #2d6a4f, #52b788);
    transition: width 0.8s ease;
}

/* ── Metric pills ── */
.metric-row { display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0; }
.metric-pill {
    background: #0d2818;
    border: 1px solid #2d6a4f44;
    border-radius: 10px;
    padding: 12px 18px;
    flex: 1;
    min-width: 110px;
    text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--green-bright);
    display: block;
}
.metric-label {
    font-size: 0.72rem;
    color: #95d5b288;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 2px;
}

/* ── Advisory box ── */
.advisory-box {
    background: linear-gradient(145deg, #0f2a1a, #0a1f12);
    border-left: 4px solid var(--green-bright);
    border-radius: 0 12px 12px 0;
    padding: 20px 24px;
    margin: 12px 0;
}
.advisory-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: var(--green-bright);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 6px;
}
.advisory-text {
    font-size: 0.95rem;
    color: var(--cream);
    line-height: 1.65;
}

/* ── Ontology tag ── */
.ontology-tag {
    display: inline-block;
    background: #e9a82511;
    border: 1px solid #e9a82544;
    color: var(--amber);
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    padding: 5px 14px;
    border-radius: 20px;
    margin: 4px 2px;
    letter-spacing: 0.5px;
}

/* ── Section divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #2d6a4f55, transparent);
    margin: 28px 0;
}

/* ── Upload zone override ── */
[data-testid="stFileUploader"] {
    background: #0f2216;
    border: 2px dashed #2d6a4f88;
    border-radius: 14px;
    padding: 20px;
}

/* ── Streamlit buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #2d6a4f, #52b788);
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    padding: 10px 28px;
    font-size: 0.95rem;
    cursor: pointer;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1f12 0%, #0d2818 100%);
    border-right: 1px solid #2d6a4f33;
}
[data-testid="stSidebar"] .css-1d391kg { background: transparent; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0d2818;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #95d5b2;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #2d6a4f;
    color: white;
}

/* ── Status chip ── */
.chip-healthy {
    background: #52b78822; border: 1px solid #52b78866;
    color: #95d5b2; border-radius: 20px;
    padding: 3px 14px; font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
    display: inline-block;
}
.chip-disease {
    background: #c1440e22; border: 1px solid #c1440e66;
    color: #ff9b77; border-radius: 20px;
    padding: 3px 14px; font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
    display: inline-block;
}

/* ── Info table rows ── */
.info-row {
    display: flex;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid #2d6a4f22;
    align-items: flex-start;
}
.info-key {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: var(--green-bright);
    text-transform: uppercase;
    letter-spacing: 1px;
    min-width: 130px;
    padding-top: 2px;
}
.info-val {
    font-size: 0.92rem;
    color: var(--cream);
    line-height: 1.5;
}

/* ── Spinner override ── */
.stSpinner > div > div { border-top-color: #52b788 !important; }

/* ── selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: #0f2216;
    border: 1px solid #2d6a4f66;
    border-radius: 10px;
    color: var(--cream);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CLASS_JSON  = os.path.join(BASE_DIR, "class_names.json")
EFF_PATH    = os.path.join(BASE_DIR, "efficientnet_best.pth")
VIT_PATH    = os.path.join(BASE_DIR, "vit_best.pth")
NUM_CLASSES = 15

AGROVOC = {
    "Pepper__bell___Bacterial_spot":               "AGROVOC c_72352 — Bacterial spot (Xanthomonas campestris)",
    "Pepper__bell___healthy":                      "AGROVOC c_5873  — Healthy plant tissue",
    "Potato___Early_blight":                       "AGROVOC c_25158 — Early blight (Alternaria solani)",
    "Potato___Late_blight":                        "AGROVOC c_25159 — Late blight (Phytophthora infestans)",
    "Potato___healthy":                            "AGROVOC c_5873  — Healthy plant tissue",
    "Tomato_Bacterial_spot":                       "AGROVOC c_72352 — Bacterial spot (Xanthomonas vesicatoria)",
    "Tomato_Early_blight":                         "AGROVOC c_25158 — Early blight (Alternaria solani)",
    "Tomato_Late_blight":                          "AGROVOC c_25159 — Late blight (Phytophthora infestans)",
    "Tomato_Leaf_Mold":                            "AGROVOC c_4871  — Leaf mold (Passalora fulva)",
    "Tomato_Septoria_leaf_spot":                   "AGROVOC c_6925  — Septoria leaf spot (Septoria lycopersici)",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "AGROVOC c_7345  — Spider mite (Tetranychus urticae)",
    "Tomato__Target_Spot":                         "AGROVOC c_4871  — Target spot (Corynespora cassiicola)",
    "Tomato__Tomato_YellowLeaf__Curl_Virus":       "AGROVOC c_92411 — Tomato yellow leaf curl virus (Begomovirus)",
    "Tomato__Tomato_mosaic_virus":                 "AGROVOC c_92412 — Tomato mosaic virus (Tobamovirus)",
    "Tomato_healthy":                              "AGROVOC c_5873  — Healthy plant tissue",
}

DISEASE_DB = {
    "Pepper__bell___Bacterial_spot":               {"cause": "Xanthomonas campestris bacteria.",         "remedy": "Remove infected leaves. Apply copper-based spray.",     "prevention": "Use resistant varieties. Avoid overhead irrigation.", "severity": "Medium"},
    "Pepper__bell___healthy":                      {"cause": "No disease detected.",                     "remedy": "Maintain balanced watering and nutrition.",             "prevention": "Routine monitoring and farm hygiene.",               "severity": "None"},
    "Potato___Early_blight":                       {"cause": "Alternaria solani fungus.",                 "remedy": "Neem oil spray. Remove lower infected leaves.",         "prevention": "Crop rotation. Proper plant spacing.",               "severity": "Medium"},
    "Potato___Late_blight":                        {"cause": "Phytophthora infestans pathogen.",          "remedy": "Apply copper or mancozeb fungicide immediately.",       "prevention": "Plant certified disease-free seed potatoes.",        "severity": "High"},
    "Potato___healthy":                            {"cause": "No disease detected.",                     "remedy": "Maintain soil fertility with compost.",                "prevention": "Regular field monitoring.",                          "severity": "None"},
    "Tomato_Bacterial_spot":                       {"cause": "Xanthomonas vesicatoria bacteria.",        "remedy": "Copper hydroxide spray. Remove plant debris.",          "prevention": "Use disease-free seed. Avoid leaf wetness.",         "severity": "Medium"},
    "Tomato_Early_blight":                         {"cause": "Alternaria solani fungus.",                 "remedy": "Neem oil or chlorothalonil fungicide.",                "prevention": "Crop rotation. Remove infected leaf debris.",        "severity": "Medium"},
    "Tomato_Late_blight":                          {"cause": "Phytophthora infestans fungus.",            "remedy": "Apply metalaxyl fungicide. Remove infected tissue.",   "prevention": "Avoid humidity. Plant resistant varieties.",         "severity": "High"},
    "Tomato_Leaf_Mold":                            {"cause": "Passalora fulva fungus (high humidity).",  "remedy": "Improve airflow. Apply chlorothalonil fungicide.",      "prevention": "Maintain low humidity. Prune for airflow.",         "severity": "Medium"},
    "Tomato_Septoria_leaf_spot":                   {"cause": "Septoria lycopersici fungus.",             "remedy": "Remove infected leaves. Apply neem oil.",              "prevention": "Avoid overhead watering. Mulch soil.",              "severity": "Medium"},
    "Tomato_Spider_mites_Two_spotted_spider_mite": {"cause": "Tetranychus urticae mite infestation.",    "remedy": "Insecticidal soap or abamectin miticide.",             "prevention": "Control humidity. Introduce predatory mites.",      "severity": "Medium"},
    "Tomato__Target_Spot":                         {"cause": "Corynespora cassiicola fungus.",           "remedy": "Remove affected leaves. Apply azoxystrobin.",          "prevention": "Field sanitation. Avoid dense planting.",           "severity": "Medium"},
    "Tomato__Tomato_YellowLeaf__Curl_Virus":       {"cause": "Begomovirus transmitted via whiteflies.",  "remedy": "Control whiteflies with yellow sticky traps.",         "prevention": "Use resistant varieties. Install reflective mulch.","severity": "High"},
    "Tomato__Tomato_mosaic_virus":                 {"cause": "Tobamovirus spread by contact.",           "remedy": "Remove and destroy infected plants immediately.",      "prevention": "Sanitize tools with bleach. Wash hands.",           "severity": "High"},
    "Tomato_healthy":                              {"cause": "No disease detected.",                     "remedy": "Balanced NPK fertilization.",                         "prevention": "Regular inspection. Maintain good airflow.",        "severity": "None"},
}

SEVERITY_COLOR = {"None": "#52b788", "Medium": "#e9a825", "High": "#c1440e"}
SEVERITY_EMOJI = {"None": "✅", "Medium": "⚠️", "High": "🚨"}


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def clean_name(raw):
    return raw.replace("___", " ").replace("__", " ").replace("_", " ").strip()

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ─────────────────────────────────────────────
#  TRANSFORMS
# ─────────────────────────────────────────────
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ─────────────────────────────────────────────
#  CACHED LOADERS
# ─────────────────────────────────────────────
@st.cache_resource
def load_class_names():
    if os.path.exists(CLASS_JSON):
        with open(CLASS_JSON) as f:
            return json.load(f)
    return list(DISEASE_DB.keys())

@st.cache_resource
def load_efficientnet(num_classes):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    if os.path.exists(EFF_PATH):
        model.load_state_dict(torch.load(EFF_PATH, map_location="cpu", weights_only=False))
    else:
        st.warning(f"EfficientNet weights not found at: {EFF_PATH}")
    model.eval()
    return model

@st.cache_resource
def load_vit(num_classes):
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
    if os.path.exists(VIT_PATH):
        model.load_state_dict(torch.load(VIT_PATH, map_location="cpu", weights_only=False))
    else:
        st.warning(f"ViT weights not found at: {VIT_PATH}")
    model.eval()
    return model

@st.cache_resource
def load_llm():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    model     = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    model.eval()
    return tokenizer, model


# ─────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────
def predict(model, image: Image.Image, class_names):
    tensor = val_transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[0]
    top5_probs, top5_idx = torch.topk(probs, min(5, len(class_names)))
    top_class = class_names[top5_idx[0].item()]
    confidence = top5_probs[0].item() * 100
    top5 = [(class_names[i], p.item() * 100) for i, p in zip(top5_idx, top5_probs)]
    return top_class, confidence, top5, probs.numpy()


# ─────────────────────────────────────────────
#  CHART BUILDERS
# ─────────────────────────────────────────────
BG  = "#0a1f12"
BG2 = "#0d2818"
FG  = "#f8f4e8"
GRN = "#52b788"
AMB = "#e9a825"
RED = "#c1440e"

def make_top5_chart(top5, model_name):
    labels = [clean_name(c) for c, _ in top5]
    values = [p for _, p in top5]
    colors = [GRN if i == 0 else "#2d6a4f" for i in range(len(values))]

    fig, ax = plt.subplots(figsize=(6, 3.2), facecolor=BG)
    ax.set_facecolor(BG)
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1],
                   edgecolor="none", height=0.55)
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va='center', ha='left',
                color=FG, fontsize=9, fontfamily='monospace')
    ax.set_xlim(0, 110)
    ax.set_xlabel("Confidence (%)", color="#95d5b2", fontsize=9)
    ax.set_title(f"{model_name} — Top-5 Predictions", color=FG,
                 fontsize=10, fontweight='bold', pad=10)
    ax.tick_params(colors=FG, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d6a4f33")
    ax.xaxis.label.set_color("#95d5b2")
    plt.tight_layout()
    return fig

def make_comparison_chart(eff_metrics, vit_metrics):
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x = np.arange(len(metrics))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG)
    ax.set_facecolor(BG)

    b1 = ax.bar(x - w/2, eff_metrics, w, label='EfficientNet-B0',
                color="#2d6a4f", edgecolor="#52b78844", linewidth=0.8)
    b2 = ax.bar(x + w/2, vit_metrics, w, label='Vision Transformer (ViT)',
                color="#1a4060", edgecolor="#4da8da44", linewidth=0.8)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.annotate(f'{h:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8,
                    color=FG, fontfamily='monospace')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color=FG, fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score", color="#95d5b2", fontsize=9)
    ax.set_title("Model Performance Comparison", color=FG, fontsize=11, fontweight='bold', pad=12)
    ax.legend(fontsize=8, facecolor=BG2, edgecolor="#2d6a4f55", labelcolor=FG)
    ax.tick_params(axis='y', colors=FG, labelsize=8)
    ax.grid(axis='y', color="#2d6a4f33", linestyle='--', linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d6a4f33")
    plt.tight_layout()
    return fig

def make_prob_radar(probs, class_names):
    # Donut chart of top-8 classes
    top_idx  = np.argsort(probs)[::-1][:8]
    top_vals = probs[top_idx]
    top_lbls = [clean_name(class_names[i])[:20] for i in top_idx]
    rest_val = 1 - top_vals.sum()
    if rest_val > 0:
        top_vals = np.append(top_vals, rest_val)
        top_lbls.append("Others")

    palette = [GRN, "#2d6a4f", "#1a4731", "#134025", "#0f3020",
               "#0a2018", "#071812", "#041008", "#e9a82544"]
    colors  = palette[:len(top_vals)]

    fig, ax = plt.subplots(figsize=(5, 4), facecolor=BG)
    ax.set_facecolor(BG)
    wedges, _ = ax.pie(top_vals, colors=colors, startangle=90,
                       wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=1.5))
    ax.text(0, 0, f"{probs.max()*100:.0f}%\nconf",
            ha='center', va='center', fontsize=12,
            color=GRN, fontfamily='monospace', fontweight='bold')

    legend = ax.legend(wedges, top_lbls, loc="lower center",
                       bbox_to_anchor=(0.5, -0.28),
                       ncol=2, fontsize=7,
                       facecolor=BG2, edgecolor="#2d6a4f55", labelcolor=FG)
    ax.set_title("Probability Distribution", color=FG, fontsize=10, fontweight='bold', pad=8)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 20px 0 10px 0;'>
        <div style='font-family: DM Serif Display, serif; font-size: 1.6rem; color: #f8f4e8;'>🌿 PlantGuard</div>
        <div style='font-size: 0.78rem; color: #52b788; font-family: Space Mono, monospace; letter-spacing: 1px; margin-top: 4px;'>AI DISEASE ADVISOR</div>
    </div>
    <hr style='border-color: #2d6a4f33; margin: 12px 0 20px 0;'>
    """, unsafe_allow_html=True)

    st.markdown("<div style='color: #95d5b2; font-size: 0.8rem; font-weight: 600; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;'>Model Selection</div>", unsafe_allow_html=True)
    selected_model = st.selectbox("Model", ["EfficientNet-B0", "Vision Transformer (ViT)", "Both Models"], label_visibility="collapsed")

    st.markdown("<hr style='border-color: #2d6a4f22; margin: 20px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='color: #95d5b2; font-size: 0.8rem; font-weight: 600; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;'>Options</div>", unsafe_allow_html=True)
    use_llm    = st.toggle("🤖 LLM Advisory (Flan-T5)", value=True)
    show_topk  = st.toggle("📊 Show Top-5 Chart",        value=True)
    show_donut = st.toggle("🍩 Show Probability Donut",   value=True)

    st.markdown("<hr style='border-color: #2d6a4f22; margin: 20px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size: 0.78rem; color: #52b78888; line-height: 1.6;'>
        <b style='color: #52b788;'>Dataset:</b> PlantVillage<br>
        <b style='color: #52b788;'>Classes:</b> 15<br>
        <b style='color: #52b788;'>Models:</b> EfficientNet-B0, ViT-B/16<br>
        <b style='color: #52b788;'>LLM:</b> Google Flan-T5<br>
        <b style='color: #52b788;'>Ontology:</b> AGROVOC (FAO)<br>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD RESOURCES
# ─────────────────────────────────────────────
class_names = load_class_names()
num_classes = len(class_names)

with st.spinner("Loading models..."):
    eff_model = load_efficientnet(num_classes)
    vit_model = load_vit(num_classes)

if use_llm:
    with st.spinner("Loading Flan-T5 LLM..."):
        llm = load_llm()


# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">COMPUTER VISION · DEEP LEARNING · NLP</div>
    <div class="hero-title">Plant<span>Guard</span> AI</div>
    <div class="hero-subtitle">
        Transfer learning plant disease detection &amp; intelligent agricultural advisory system.<br>
        Powered by EfficientNet-B0, Vision Transformer, and Flan-T5 LLM with AGROVOC ontology mapping.
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔬  Diagnose Plant", "📊  Model Comparison", "📖  Disease Database"])


# ════════════════════════════════════════════
#  TAB 1 — DIAGNOSE
# ════════════════════════════════════════════
with tab1:
    col_upload, col_result = st.columns([1, 1.4], gap="large")

    with col_upload:
        st.markdown('<div class="card"><div class="card-title"><span>📷</span>Upload Leaf Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"],
                                    help="Upload a clear leaf image for disease diagnosis",
                                    label_visibility="collapsed")

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, use_container_width=True,
                     caption="Uploaded Image", output_format="PNG")

        st.markdown('</div>', unsafe_allow_html=True)

        # Quick demo picker
        st.markdown('<div class="card"><div class="card-title"><span>💡</span>Quick Demo Classes</div>', unsafe_allow_html=True)
        demo_class = st.selectbox("Select a class to preview", ["—"] + [clean_name(c) for c in class_names], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        if uploaded is None:
            st.markdown("""
            <div style='display:flex; flex-direction:column; align-items:center;
                        justify-content:center; height:400px; color:#2d6a4f88;
                        text-align:center; border: 2px dashed #2d6a4f33; border-radius:16px;'>
                <div style='font-size:4rem; margin-bottom:16px;'>🌱</div>
                <div style='font-family: DM Serif Display, serif; font-size:1.4rem; color:#52b78888;'>
                    Upload a leaf image to begin diagnosis
                </div>
                <div style='font-size:0.85rem; color:#2d6a4f88; margin-top:8px;'>
                    Supports JPG, JPEG, PNG
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("🔬 Analysing leaf image..."):
                # Run selected model(s)
                if selected_model == "EfficientNet-B0":
                    pred_class, conf, top5, all_probs = predict(eff_model, image, class_names)
                    model_used = "EfficientNet-B0"
                elif selected_model == "Vision Transformer (ViT)":
                    pred_class, conf, top5, all_probs = predict(vit_model, image, class_names)
                    model_used = "Vision Transformer (ViT)"
                else:
                    pred_class, conf, top5, all_probs = predict(eff_model, image, class_names)
                    vit_pred, vit_conf, vit_top5, vit_probs = predict(vit_model, image, class_names)
                    model_used = "Both Models"

            info       = DISEASE_DB.get(pred_class, {})
            severity   = info.get("severity", "Unknown")
            is_healthy = "healthy" in pred_class.lower()
            card_cls   = "prediction-healthy" if is_healthy else "prediction-disease"
            sev_color  = SEVERITY_COLOR.get(severity, "#888")
            sev_emoji  = SEVERITY_EMOJI.get(severity, "❓")
            chip_cls   = "chip-healthy" if is_healthy else "chip-disease"

            # ── Prediction card ──
            st.markdown(f"""
            <div class="{card_cls}">
                <div class="{chip_cls}">{sev_emoji} {'HEALTHY' if is_healthy else 'DISEASE DETECTED'}</div>
                <div class="prediction-label">{clean_name(pred_class)}</div>
                <div class="prediction-sub">MODEL: {model_used.upper()}</div>
                <div style='margin: 16px 0 8px 0;'>
                    <div style='display:flex; justify-content:space-between;
                                font-size:0.8rem; color:#95d5b2; margin-bottom:4px;'>
                        <span>Confidence</span><span style='font-family:monospace; color:#52b788;'>{conf:.1f}%</span>
                    </div>
                    <div class="conf-bar-wrap">
                        <div class="conf-bar-fill" style='width:{conf}%; background:{"linear-gradient(90deg,#2d6a4f,#52b788)" if is_healthy else "linear-gradient(90deg,#8b2500,#c1440e)"}'></div>
                    </div>
                </div>
                <div style='margin-top:8px;'>
                    <span class="ontology-tag">{AGROVOC.get(pred_class, 'N/A')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Both models comparison ──
            if selected_model == "Both Models":
                st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
                ca, cb = st.columns(2)
                with ca:
                    vit_match = "✅ Agreement" if pred_class == vit_pred else "⚠️ Disagreement"
                    st.markdown(f"""
                    <div class="metric-pill" style='text-align:left; padding:14px 16px;'>
                        <div class="metric-label">EfficientNet Confidence</div>
                        <div class="metric-value">{conf:.1f}%</div>
                    </div>""", unsafe_allow_html=True)
                with cb:
                    st.markdown(f"""
                    <div class="metric-pill" style='text-align:left; padding:14px 16px;'>
                        <div class="metric-label">ViT Confidence</div>
                        <div class="metric-value">{vit_conf:.1f}%</div>
                    </div>""", unsafe_allow_html=True)
                st.markdown(f"""<div style='text-align:center; font-family:Space Mono, monospace;
                    font-size:0.8rem; color:#95d5b2; margin: 8px 0;'>{vit_match}</div>""",
                    unsafe_allow_html=True)

            # ── Disease details ──
            if info:
                st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="card">
                    <div class="card-title"><span>🧬</span>Disease Details</div>
                    <div class="info-row">
                        <div class="info-key">Cause</div>
                        <div class="info-val">{info.get('cause','—')}</div>
                    </div>
                    <div class="info-row">
                        <div class="info-key">Remedy</div>
                        <div class="info-val">{info.get('remedy','—')}</div>
                    </div>
                    <div class="info-row">
                        <div class="info-key">Prevention</div>
                        <div class="info-val">{info.get('prevention','—')}</div>
                    </div>
                    <div class="info-row" style='border:none;'>
                        <div class="info-key">Severity</div>
                        <div class="info-val" style='color:{sev_color}; font-weight:600;'>{sev_emoji} {severity}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── LLM Advisory ──
            if use_llm and not is_healthy:
                with st.spinner("🤖 Generating LLM advisory..."):
                    prompt = (
                        f"Disease: {clean_name(pred_class)}. "
                        f"Cause: {info.get('cause','')} "
                        f"Remedy: {info.get('remedy','')} "
                        f"Prevention: {info.get('prevention','')} "
                        f"Write a short helpful advisory for a farmer."
                    )
                    tokenizer, llm_model = llm
                    inputs   = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
                    with torch.no_grad():
                        outputs = llm_model.generate(**inputs, max_new_tokens=150, num_beams=4, early_stopping=True)
                    llm_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                st.markdown(f"""
                <div class="advisory-box">
                    <div class="advisory-label">🤖 Flan-T5 LLM Advisory</div>
                    <div class="advisory-text">{llm_text}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Charts row ──
    if uploaded:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        chart_cols = []
        if show_topk:    chart_cols.append("top5")
        if show_donut:   chart_cols.append("donut")
        if selected_model == "Both Models": chart_cols.append("vit_top5")

        if chart_cols:
            cols = st.columns(len(chart_cols))
            idx  = 0
            if show_topk and idx < len(cols):
                fig = make_top5_chart(top5, "EfficientNet-B0")
                cols[idx].pyplot(fig, use_container_width=True)
                plt.close(fig)
                idx += 1
            if show_donut and idx < len(cols):
                fig = make_prob_radar(all_probs, class_names)
                cols[idx].pyplot(fig, use_container_width=True)
                plt.close(fig)
                idx += 1
            if selected_model == "Both Models" and idx < len(cols):
                fig = make_top5_chart(vit_top5, "Vision Transformer (ViT)")
                cols[idx].pyplot(fig, use_container_width=True)
                plt.close(fig)


# ════════════════════════════════════════════
#  TAB 2 — MODEL COMPARISON
# ════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="card">
        <div class="card-title"><span>📊</span>Model Performance Dashboard</div>
        <div style='font-size:0.88rem; color:#95d5b288; line-height:1.6;'>
            Performance metrics after training on PlantVillage dataset (70% train / 15% val / 15% test split).
            Enter your actual results from notebook training below.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='color: #95d5b2; font-size:0.85rem; font-weight:600; margin-bottom:8px;'>Enter Model Results</div>", unsafe_allow_html=True)

    mc1, mc2 = st.columns(2, gap="large")

    with mc1:
        st.markdown('<div class="card"><div class="card-title"><span>🔵</span>EfficientNet-B0</div>', unsafe_allow_html=True)
        eff_acc  = st.slider("Accuracy",  0.0, 1.0, 0.94, 0.001, key="ea")
        eff_prec = st.slider("Precision", 0.0, 1.0, 0.94, 0.001, key="ep")
        eff_rec  = st.slider("Recall",    0.0, 1.0, 0.93, 0.001, key="er")
        eff_f1   = st.slider("F1-Score",  0.0, 1.0, 0.93, 0.001, key="ef")
        st.markdown('</div>', unsafe_allow_html=True)

    with mc2:
        st.markdown('<div class="card"><div class="card-title"><span>🟠</span>Vision Transformer (ViT)</div>', unsafe_allow_html=True)
        vit_acc  = st.slider("Accuracy",  0.0, 1.0, 0.92, 0.001, key="va")
        vit_prec = st.slider("Precision", 0.0, 1.0, 0.92, 0.001, key="vp")
        vit_rec  = st.slider("Recall",    0.0, 1.0, 0.91, 0.001, key="vr")
        vit_f1   = st.slider("F1-Score",  0.0, 1.0, 0.91, 0.001, key="vf")
        st.markdown('</div>', unsafe_allow_html=True)

    # Summary metrics
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)
    pill_data = [
        (m1, eff_acc,  "Eff Acc"),  (m2, eff_prec, "Eff Prec"),
        (m3, eff_rec,  "Eff Recall"),(m4, eff_f1,   "Eff F1"),
        (m5, vit_acc,  "ViT Acc"),  (m6, vit_prec, "ViT Prec"),
        (m7, vit_rec,  "ViT Recall"),(m8, vit_f1,   "ViT F1"),
    ]
    for col, val, lbl in pill_data:
        col.markdown(f"""
        <div class="metric-pill">
            <span class="metric-value">{val:.2f}</span>
            <div class="metric-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    # Comparison chart
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    fig = make_comparison_chart(
        [eff_acc, eff_prec, eff_rec, eff_f1],
        [vit_acc, vit_prec, vit_rec, vit_f1]
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Observations
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <div class="card-title"><span>📝</span>Observations & Justifications</div>
        <div class="info-row">
            <div class="info-key">EfficientNet</div>
            <div class="info-val">Compound scaling (~5.3M params) excels at fine-grained texture differences between disease classes. Faster convergence due to efficient architecture.</div>
        </div>
        <div class="info-row">
            <div class="info-key">ViT-B/16</div>
            <div class="info-val">Self-attention captures global leaf patterns (mosaic virus spread, full-leaf yellowing) that CNN local filters can miss. Requires more epochs to converge.</div>
        </div>
        <div class="info-row">
            <div class="info-key">Preprocessing</div>
            <div class="info-val">RandomFlip, Rotation, ColorJitter simulate real outdoor field photography. ImageNet normalization required for pretrained weight compatibility.</div>
        </div>
        <div class="info-row">
            <div class="info-key">Transfer Learning</div>
            <div class="info-val">Both models use pretrained ImageNet weights, reducing training time and preventing overfitting on the relatively small PlantVillage dataset.</div>
        </div>
        <div class="info-row" style='border:none;'>
            <div class="info-key">Early Stopping</div>
            <div class="info-val">Patience=3 stops training when val loss plateaus, saving best weights. ReduceLROnPlateau allows finer gradient updates in later training stages.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════
#  TAB 3 — DISEASE DATABASE
# ════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="card">
        <div class="card-title"><span>📖</span>Agricultural Disease Knowledge Base</div>
        <div style='font-size:0.88rem; color:#95d5b288;'>
            15 plant disease classes with structured information and AGROVOC (FAO) ontology cross-references.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Filter
    filter_col1, filter_col2 = st.columns([2, 1])
    with filter_col1:
        search = st.text_input("🔍 Search disease or plant", "", placeholder="e.g. Tomato, blight, mite...", label_visibility="collapsed")
    with filter_col2:
        sev_filter = st.selectbox("Severity", ["All", "None", "Medium", "High"], label_visibility="collapsed")

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    shown = 0
    for cls, info in DISEASE_DB.items():
        name     = clean_name(cls)
        severity = info["severity"]
        ontology = AGROVOC.get(cls, "")

        if search.lower() and search.lower() not in name.lower() and search.lower() not in ontology.lower():
            continue
        if sev_filter != "All" and severity != sev_filter:
            continue

        sev_color = SEVERITY_COLOR.get(severity, "#888")
        sev_emoji = SEVERITY_EMOJI.get(severity, "❓")
        is_h      = "healthy" in cls.lower()
        chip      = "chip-healthy" if is_h else "chip-disease"

        with st.expander(f"{sev_emoji}  {name}", expanded=False):
            st.markdown(f"""
            <div style='margin-bottom:10px;'>
                <span class="{chip}">{'HEALTHY' if is_h else 'DISEASE'}</span>
                &nbsp;
                <span style='color:{sev_color}; font-family:Space Mono,monospace;
                             font-size:0.75rem;'>SEVERITY: {severity.upper()}</span>
            </div>
            <div class="ontology-tag">{ontology}</div>
            <div style='height:12px;'></div>
            <div class="info-row">
                <div class="info-key">Cause</div>
                <div class="info-val">{info['cause']}</div>
            </div>
            <div class="info-row">
                <div class="info-key">Remedy</div>
                <div class="info-val">{info['remedy']}</div>
            </div>
            <div class="info-row" style='border:none;'>
                <div class="info-key">Prevention</div>
                <div class="info-val">{info['prevention']}</div>
            </div>
            """, unsafe_allow_html=True)
        shown += 1

    if shown == 0:
        st.markdown("""
        <div style='text-align:center; padding:40px; color:#2d6a4f88;'>
            No diseases found matching your search.
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='text-align:center; font-family:Space Mono,monospace;
                font-size:0.72rem; color:#2d6a4f88; margin-top:20px;'>
        Showing {shown} of {len(DISEASE_DB)} classes &nbsp;·&nbsp;
        Ontology: AGROVOC (FAO) &nbsp;·&nbsp; Dataset: PlantVillage
    </div>
    """, unsafe_allow_html=True)