import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
import speech_recognition as sr
import io
import smtplib
import ssl
from email.message import EmailMessage
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardcoded Gmail credentials
SENDER_EMAIL = "madhuchillar01@gmail.com"
SENDER_PASSWORD = "zhkrzejamijkmavk"

# Streamlit page config
st.set_page_config(page_title="🎤 AI Speech-to-Image Generator", layout="centered", page_icon="🎨")

# Custom CSS
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #e3eafc, #f9f9ff);
    }
    .stButton>button {
        border-radius: 12px;
        background-color: #6c63ff;
        color: white;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: 0.4s ease;
    }
    .stButton>button:hover {
        background-color: #4a47a3;
        transform: translateY(-2px) scale(1.02);
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
        padding: 0.5rem;
        background-color: #f2f3f5;
    }
    footer {
        text-align: center;
        color: gray;
        font-size: 0.9rem;
        padding: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎨 AI Speech-to-Image Generator")
st.markdown("Turn your **voice** or **text** into precise AI images using **Stable Diffusion v1.5**.")

# Load the Stable Diffusion pipeline
@st.cache_resource
def load_pipeline():
    start_time = time.time()
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            "./stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        pipe.to("cuda")
        pipe.enable_model_cpu_offload()  # Manage VRAM for RTX 2050
        logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
        return pipe
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        st.error(f"❌ Failed to load model: {e}")
        return None

pipe = load_pipeline()

if pipe is None:
    st.error("❌ Model loading failed. Ensure './stable-diffusion-v1-5' exists.")
    st.stop()

# Helper functions
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎧 Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        st.info("🎙️ Listening...")
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=20)
            st.info("🧠 Recognizing speech...")
            text = recognizer.recognize_google(audio)
            st.success(f"✅ Recognized: {text}")
            return text
        except sr.UnknownValueError:
            st.error("❌ Could not understand audio.")
        except sr.RequestError as e:
            st.error(f"❌ Error: {e}")
        except sr.WaitTimeoutError:
            st.error("❌ Listening timed out, please try again.")
    return None

def generate_image(prompt, style=None, negative_prompt=None):
    start_time = time.time()
    try:
        torch.cuda.empty_cache()
        with torch.no_grad():
            with torch.autocast("cuda"):
                if style and style != "None":
                    prompt = f"{prompt}, {style} style"
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=20,  # Reduced for speed
                    guidance_scale=8.0,  # Higher for prompt adherence
                    height=512,
                    width=512
                )
        image = result.images[0]
        logger.info(f"Image generated in {time.time() - start_time:.2f} seconds")
        return image
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        st.error(f"❌ Image generation failed: {e}")
        return None
    finally:
        torch.cuda.empty_cache()

def compress_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", quality=85)
    buffer.seek(0)
    return buffer

def send_email_with_image(receiver_email, image_buffer):
    try:
        msg = EmailMessage()
        msg["Subject"] = "Your AI Generated Image 🎨"
        msg["From"] = SENDER_EMAIL
        msg["To"] = receiver_email
        msg.set_content("Hi there! Please find your AI-generated image attached. ✨")

        image_data = image_buffer.getvalue()
        msg.add_attachment(image_data, maintype="image", subtype="png", filename="ai_image.png")

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)

        st.success("✅ Email sent successfully!")
    except Exception as e:
        st.error(f"❌ Failed to send email: {e}")

# State variables for persistent image display
if "generated_image" not in st.session_state:
    st.session_state.generated_image = None
    st.session_state.compressed_buffer = None

# Prompt Input Section
tab1, tab2 = st.tabs(["📝 Text Prompt", "🎤 Voice Prompt"])

# --- Tab 1: Text Prompt ---
with tab1:
    prompt_text = st.text_input("Enter a prompt (e.g., a futuristic city with flying cars):", key="text_prompt_input")
    style = st.selectbox("Select Style (optional)", ["None", "Realistic", "Cartoon", "Anime", "Oil Painting"], key="text_style_selectbox")
    negative_prompt = st.text_input("Negative Prompt (optional, e.g., blurry, low quality):", key="text_negative_prompt_input")
    
    if st.button("🎨 Generate Image from Text", key="text_generate_button"):
        if prompt_text:
            with st.spinner("🎨 Generating your AI Art..."):
                st.session_state.generated_image = generate_image(prompt_text, style, negative_prompt)
                if st.session_state.generated_image:
                    st.session_state.compressed_buffer = compress_image(st.session_state.generated_image)
                    st.image(st.session_state.generated_image, caption="🖼️ Your AI Art", use_container_width=True)
        else:
            st.warning("⚠️ Please enter a prompt first.")

# --- Tab 2: Voice Prompt ---
with tab2:
    style = st.selectbox("Select Style (optional)", ["None", "Realistic", "Cartoon", "Anime", "Oil Painting"], key="voice_style_selectbox")
    negative_prompt = st.text_input("Negative Prompt (optional, e.g., blurry, low quality):", key="voice_negative_prompt_input")
    if st.button("🎤 Speak to Generate Image", key="voice_generate_button"):
        spoken_text = recognize_speech()
        if spoken_text:
            with st.spinner("🎨 Generating your AI Art..."):
                st.session_state.generated_image = generate_image(spoken_text, style, negative_prompt)
                if st.session_state.generated_image:
                    st.session_state.compressed_buffer = compress_image(st.session_state.generated_image)
                    st.image(st.session_state.generated_image, caption="🖼️ Your AI Art", use_container_width=True)

# --- Email Section ---
if st.session_state.generated_image:
    st.markdown("---")
    st.subheader("📧 Send Generated Image via Email")

    receiver_email = st.text_input("Recipient's Email", placeholder="recipient@example.com", key="receiver_email_input")

    if st.button("📤 Send Image", key="send_email_button"):
        if receiver_email:
            send_email_with_image(receiver_email, st.session_state.compressed_buffer)
        else:
            st.warning("⚠️ Please enter a recipient's email and generate an image first.")

    # Download button
    st.download_button(
        label="⬇️ Download Image",
        data=st.session_state.compressed_buffer,
        file_name="generated_image.png",
        mime="image/png",
        key="download_image_button"
    )

# Footer
st.markdown("""<footer>Made with ❤️ using Streamlit, Stable Diffusion v1.5 & creativity, by Rakshi</footer>""", unsafe_allow_html=True)
#python -m streamlit run app.py
