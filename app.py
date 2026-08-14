import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Local Image AI",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 Local Image AI")

st.write("Image-to-image generation interface")

uploaded_images = st.file_uploader(
    "Upload reference images",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True
)

prompt = st.text_area(
    "Prompt",
    placeholder="Describe the image you want to create..."
)

negative_prompt = st.text_area(
    "Negative Prompt",
    value="blurry, low quality, distorted"
)

if uploaded_images:

    st.subheader("Reference Images")

    columns = st.columns(
        min(len(uploaded_images), 4)
    )

    for i, uploaded_file in enumerate(uploaded_images):

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        columns[i % len(columns)].image(
            image,
            caption=f"Reference {i + 1}",
            use_container_width=True
        )


if st.button(
    "🚀 Generate",
    type="primary",
    use_container_width=True
):

    if not prompt.strip():

        st.error("Please enter a prompt.")

    elif not uploaded_images:

        st.error("Please upload at least one reference image.")

    else:

        st.success(
            "Interface is working. "
            "The local diffusion model will be connected next."
        )