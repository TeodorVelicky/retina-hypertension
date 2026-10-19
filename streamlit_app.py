# Retinal hypertension signs: research demo with Grad-CAM (Streamlit).
import json

import numpy as np
import streamlit as st
import torch
from PIL import Image

import model as M

st.set_page_config(page_title="Retinal hypertension signs", page_icon="👁️", layout="centered")

H = json.load(open("head.json"))
SPEC = H["spec"]
TEST_LOGITS = np.asarray(H["test_logits"], dtype=np.float32)


@st.cache_resource(show_spinner="Loading the vision model (first visit takes about a minute)…")
def load():
    torch.set_num_threads(2)
    need = sorted({M.FEATURES[n][0] for n in SPEC["features"]})
    net = M.RetinaModel(M.load_backbones(need), SPEC["features"], H["w"], H["b"], SPEC["tta"]).eval()
    return net, M.make_cam(net)


st.title("Retinal hypertension signs")
st.markdown(
    f"Frozen **DINOv2** image features at 448 px with **CLAHE** vessel enhancement, plus a logistic-regression "
    f"head, trained on ODIR-5K. Test AUROC **{SPEC['test_auroc']:.2f}** "
    f"(95% CI {SPEC['auroc_ci'][0]:.2f}–{SPEC['auroc_ci'][1]:.2f}).  \n"
    "**Research demo, not a medical device.** Upload a colour fundus photograph."
)

up = st.file_uploader("Fundus photo", type=["jpg", "jpeg", "png"])
if up is not None:
    img = Image.open(up)
    net, cam = load()
    rgb, x = M.prepare(img, SPEC["clahe"])
    with torch.no_grad():
        logit = float(net(x[None]).squeeze())
    rank = float((TEST_LOGITS < logit).mean() * 100)
    flagged = logit >= H["threshold"]
    with st.spinner("Computing Grad-CAM…"):
        overlay = M.cam_overlay(cam, x, rgb)

    c1, c2 = st.columns(2)
    c1.image(rgb, caption="Input (cropped, 448 × 448)", use_container_width=True)
    c2.image(overlay, caption="Where the model looked (Grad-CAM)", use_container_width=True)

    st.subheader(f"Score ranks above {rank:.0f}% of held-out test images")
    if flagged:
        st.warning("Above the screening threshold: worth a closer look.")
    else:
        st.success("Below the screening threshold.")
    st.caption(
        f"The threshold was chosen during development to catch at least 85% of cases. On the test set it caught "
        f"{SPEC['sensitivity'] * 100:.0f}% of hypertensive images and cleared {SPEC['specificity'] * 100:.0f}% "
        "of the others. The score is a rank, not a probability: the model's raw outputs are not calibrated."
    )
