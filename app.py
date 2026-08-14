# app.py

import os
import torch
import streamlit as st

from PIL import Image
from diffusers import AutoPipelineForImage2Image


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DTYPE = (
    torch.float16
    if DEVICE == "cuda"
    else torch.float32
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Local Image AI",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 Local Image AI Studio")

st.caption(
    "Prompt + reference images → generated image"
)

st.write(
    f"Running on: **{DEVICE.upper()}**"
)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():

    pipe = AutoPipelineForImage2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        use_safetensors=True
    )

    pipe.to(DEVICE)

    if DEVICE == "cuda":
        pipe.enable_attention_slicing()

    return pipe


# ============================================================
# LOAD MODEL
# ============================================================

with st.spinner("Loading model..."):

    pipe = load_model()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Generation Settings")

    strength = st.slider(
        "Image Strength",
        min_value=0.1,
        max_value=1.0,
        value=0.65,
        step=0.05
    )

    guidance = st.slider(
        "Prompt Guidance",
        min_value=1.0,
        max_value=15.0,
        value=7.5,
        step=0.5
    )

    steps = st.slider(
        "Inference Steps",
        min_value=5,
        max_value=50,
        value=20
    )

    seed = st.number_input(
        "Seed",
        min_value=0,
        max_value=999999999,
        value=42
    )

    width = st.selectbox(
        "Width",
        [512, 768],
        index=0
    )

    height = st.selectbox(
        "Height",
        [512, 768],
        index=0
    )


# ============================================================
# PROMPT
# ============================================================

st.subheader("Prompt")

prompt = st.text_area(
    "Describe what you want",
    height=120,
    placeholder=(
        "A cinematic realistic photograph of a futuristic "
        "city at night, dramatic lighting, detailed architecture, "
        "realistic atmosphere..."
    )
)


negative_prompt = st.text_area(
    "Negative Prompt",
    value=(
        "blurry, low quality, distorted, deformed, "
        "bad anatomy, duplicate objects, extra fingers, "
        "watermark, text, logo"
    ),
    height=80
)


# ============================================================
# REFERENCE IMAGES
# ============================================================

st.subheader("Reference Images")

uploaded_files = st.file_uploader(
    "Upload one or more reference images",
    type=[
        "png",
        "jpg",
        "jpeg",
        "webp"
    ],
    accept_multiple_files=True
)


reference_images = []


if uploaded_files:

    cols = st.columns(
        min(len(uploaded_files), 4)
    )

    for index, file in enumerate(uploaded_files):

        image = Image.open(file).convert("RGB")

        reference_images.append(image)

        cols[index % len(cols)].image(
            image,
            caption=f"Reference {index + 1}",
            use_container_width=True
        )


# ============================================================
# GENERATION
# ============================================================

st.divider()

generate = st.button(
    "🚀 Generate Image",
    type="primary",
    use_container_width=True
)


if generate:

    if not prompt.strip():

        st.error(
            "Please enter a prompt."
        )

        st.stop()


    if not reference_images:

        st.error(
            "Please upload at least one reference image."
        )

        st.stop()


    # --------------------------------------------------------
    # USE FIRST REFERENCE IMAGE
    # --------------------------------------------------------

    init_image = reference_images[0]

    init_image = init_image.resize(
        (width, height)
    )


    # --------------------------------------------------------
    # GENERATOR
    # --------------------------------------------------------

    generator = torch.Generator(
        device=DEVICE
    ).manual_seed(seed)


    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    with st.spinner(
        "Generating image... CPU generation may take several minutes."
    ):

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,

            image=init_image,

            strength=strength,

            guidance_scale=guidance,

            num_inference_steps=steps,

            generator=generator
        )


        generated_image = result.images[0]


    # ========================================================
    # OUTPUT
    # ========================================================

    st.subheader("Generated Image")

    st.image(
        generated_image,
        use_container_width=True
    )


    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(
        "outputs",
        exist_ok=True
    )


    output_path = (
        f"outputs/generated_{seed}.png"
    )


    generated_image.save(
        output_path
    )


    # ========================================================
    # DOWNLOAD
    # ========================================================

    with open(
        output_path,
        "rb"
    ) as file:

        st.download_button(
            "⬇️ Download Image",
            data=file,
            file_name="generated_image.png",
            mime="image/png",
            use_container_width=True
        )


# ============================================================
# INFORMATION
# ============================================================

with st.expander("Model Information"):

    st.write(
        f"""
        **Model:** {MODEL_ID}

        **Device:** {DEVICE}

        **Reference images:** {len(reference_images)}

        **Mode:** Image-to-Image

        **Strength:** {strength}

        **Guidance:** {guidance}

        **Steps:** {steps}

        **Seed:** {seed}
        """
    )