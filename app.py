import streamlit as st
import time
import tempfile
import os
import torch
from faster_whisper import WhisperModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from gtts import gTTS
import io

# --- Page Config & Mobile Optimization ---
st.set_page_config(page_title="Indic-English Translator", layout="wide", page_icon="🗣️")

# Custom CSS to make it touch-friendly and full-width on Xiaomi Pad 6
st.markdown("""
    <style>
        .block-container { padding-top: 2rem; max-width: 100%; }
        .stAudioInput { width: 100%; margin-bottom: 20px; }
        .chat-container { display: flex; flex-direction: column; gap: 10px; }
        .bubble { padding: 12px 16px; border-radius: 18px; max-width: 85%; word-wrap: break-word; font-size: 1.1rem; }
        .user-bubble { background-color: #dcf8c6; align-self: flex-end; border-bottom-right-radius: 4px; }
        .bot-bubble { background-color: #ffffff; align-self: flex-start; border-bottom-left-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.1); }
        .lang-tag { font-size: 0.8rem; font-weight: bold; color: #666; display: block; margin-bottom: 4px; }
        .status-box { background-color: #f0f2f6; padding: 10px; border-radius: 8px; text-align: center; font-weight: bold; color: #4CAF50; }
    </style>
""", unsafe_allow_html=True)

# --- Language Mapping ---
LANG_MAP = {
    "Tamil": "tam_Taml", "Telugu": "tel_Telu", "Kannada": "kan_Knda", 
    "Malayalam": "mal_Mlym", "English": "eng_Latn"
}
WHISPER_LANG_MAP = {
    "tamil": "Tamil", "telugu": "Telugu", "kannada": "Kannada", 
    "malayalam": "Malayalam", "english": "English"
}

# --- Model Loading (Cached for performance) ---
@st.cache_resource
def load_models():
    st.info("⏳ Loading AI models... (First run takes 2-3 mins to download)")
    try:
        # 1. Whisper STT (small model, int8 for CPU efficiency)
        whisper = WhisperModel("small", device="cpu", compute_type="int8")
        
        # 2. Translation Models (IndicTrans2)
        tokenizer_en_indic = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M")
        model_en_indic = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M")
        
        tokenizer_indic_en = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-indic-en-dist-200M")
        model_indic_en = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-indic-en-dist-200M")
        
        return whisper, (model_en_indic, tokenizer_en_indic), (model_indic_en, tokenizer_indic_en)
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

# --- Translation Logic ---
def translate_text(text, model, tokenizer, src_lang_code, tgt_lang_code):
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        tgt_lang_id = tokenizer.lang_code_to_id[tgt_lang_code]
        outputs = model.generate(**inputs, forced_bos_token_id=tgt_lang_id, max_new_tokens=200)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"[Translation Error: {e}]"

# --- TTS Logic ---
def speak_text(text, lang_code):
    try:
        lang = 'en' if lang_code == 'eng_Latn' else lang_code.split('_')[0][:2] # Fallback to English TTS for Indic if gTTS fails
        tts = gTTS(text=text, lang='en' if lang_code == 'eng_Latn' else 'en') # gTTS Indic support is limited, using English as placeholder per PDF
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None

# --- Main App Loop ---
def main():
    st.title("🗣️ South Indic <-> English Translator")
    st.markdown("Tap the mic, speak in Tamil/Telugu/Kannada/Malayalam/English, and let's chat!")
    
    # Load models
    whisper, (model_en_indic, tok_en_indic), (model_indic_en, tok_indic_en) = load_models()
    
    # Session state for chat history and active language
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "active_lang" not in st.session_state:
        st.session_state.active_lang = "English"

    # Status Indicator
    status_placeholder = st.empty()
    status_placeholder.markdown('<div class="status-box">🟢 Ready. Tap the mic to start.</div>', unsafe_allow_html=True)

    # Chat Display
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            bubble_class = "user-bubble" if msg["role"] == "user" else "bot-bubble"
            st.markdown(f'<div class="bubble {bubble_class}"><span class="lang-tag">{msg["lang"]}</span>{msg["text"]}</div>', unsafe_allow_html=True)

    # Audio Input (Optimized for touch)
    audio_value = st.audio_input("🎙️ Tap and hold to speak", key="mic_input")

    if audio_value:
        try:
            # 1. Save audio to temp file
            status_placeholder.markdown('<div class="status-box">🎙️ Listening... Processing Audio</div>', unsafe_allow_html=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_value.read())
                tmp_path = tmp.name

            # 2. Transcribe with Whisper
            status_placeholder.markdown('<div class="status-box">📝 Transcribing...</div>', unsafe_allow_html=True)
            start_time = time.time()
            segments, info = whisper.transcribe(tmp_path, beam_size=5)
            transcript = " ".join([segment.text for segment in segments]).strip()
            detected_lang = WHISPER_LANG_MAP.get(info.language.lower(), "English")
            st.session_state.active_lang = detected_lang
            print(f"STT Time: {time.time() - start_time:.2f}s | Lang: {detected_lang}")

            if not transcript:
                status_placeholder.markdown('<div class="status-box">⚠️ No speech detected. Try again.</div>', unsafe_allow_html=True)
                return

            # Add user message to chat
            st.session_state.messages.append({"role": "user", "lang": detected_lang, "text": transcript})

            # 3. Translate if needed
            if detected_lang != "English":
                status_placeholder.markdown('<div class="status-box">🔄 Translating to English...</div>', unsafe_allow_html=True)
                start_time = time.time()
                eng_text = translate_text(transcript, model_indic_en, tok_indic_en, LANG_MAP[detected_lang], "eng_Latn")
                print(f"Translation Time: {time.time() - start_time:.2f}s")
                
                st.session_state.messages.append({"role": "bot", "lang": "English (Translated)", "text": eng_text})
                
                # Speak English
                status_placeholder.markdown('<div class="status-box">🔊 Speaking English...</div>', unsafe_allow_html=True)
                audio_bytes = speak_text(eng_text, "eng_Latn")
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
            else:
                # If user spoke English, we assume they are replying to the last Indic language
                # For simplicity in this prototype, we just echo it or translate to the last known Indic lang
                st.session_state.messages.append({"role": "bot", "lang": "English", "text": transcript})
                status_placeholder.markdown('<div class="status-box">🟢 Ready.</div>', unsafe_allow_html=True)

            # Clean up temp file
            os.remove(tmp_path)
            st.rerun()

        except Exception as e:
            st.error(f"An error occurred: {e}")
            status_placeholder.markdown('<div class="status-box">❌ Error. Please try again.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
