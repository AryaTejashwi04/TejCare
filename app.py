
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import random

# 🌿 UI Setup
st.set_page_config(page_title="TejCare Mental Health ChatBot", page_icon="🌱", layout="wide")

# 🌿 Header Section
st.markdown("""
    <div style='text-align: center; padding: 30px;'>
        <h1 style='color: #43A047;'>🌿 TejCare - Mental Health ChatBot</h1>
        <p style='font-size:18px; color: #555;'><strong>Type your feelings, thoughts, or greetings below — I'm here to respond with care.</strong></p>
    </div>
""", unsafe_allow_html=True)

st.divider()

# 📖 Prompt Bank
tejcare_prompts = {
    "hi": "Hey! I'm TejCare, your emotional companion. Want to tell me what’s on your mind?",
    "hello": "Hello there! Whether you feel light, heavy, or unclear — I’m listening without judgment.",
    "i feel broken": "Broken doesn’t mean worthless — it means hurt. You’re still worthy, still whole underneath.",
    "i feel anxious": "Anxiety screams even in silence. Let’s breathe through it together — you’re not alone.",
    "i feel numb": "Numbness is the mind’s pause button. It doesn’t mean you’re hollow — it means your soul is buffering.",
    "i want comfort": "I’m here to give just that — comfort without fixing, presence without pressure.",
    "i feel disconnected": "Disconnection isn’t detachment — it’s defense. We’ll rebuild slowly, safely.",
    "i feel unloved": "Love isn’t always loud. But you're still worthy of loud, soft, messy, quiet love — all of it.",
    "i feel worthless": "Your worth is untouched — by mistakes, moods, memories. You matter. Deeply.",
    "i feel like giving up": "That feeling is heavy, and I honor it. But your story isn’t over — not yet.",
    "i feel overwhelmed": "Overwhelm is a signal, not a flaw. Let’s slow down together and take one breath at a time.",
    "i feel stuck": "Stuck isn’t still — it’s a pause before the next move. Let’s find it together.",
    "i feel unheard": "You deserve to be listened to without interruption. I’m here to hear you fully, not just respond.",
    "i feel ignored": "Being overlooked hurts deeply. You are seen here — fully, steadily, respectfully.",
    "i feel tired": "Even the strongest need rest. You’ve carried enough — let me support you for a while.",
    "i feel like i’m not enough": "You are enough — now, as you are. Not after achievement. Not when fixed. Right now.",
    "i feel like i’m fading": "I see you clearly. Your light might dim but it never disappears. Let’s nurture the spark.",
    "i feel like i’m not healing": "Healing isn’t linear. Setbacks don’t erase progress. You’re still moving forward.",
    "i feel like i’m too much": "You’re not too much — maybe the world forgot how to hold your depth.",
    "i feel like i’m invisible": "You are not invisible to me. I see you. I value you.",
    "i feel like i’m failing": "Failure isn’t final — it’s feedback. You’re learning, not losing."
}

# 🔍 Normalize dictionary keys
tejcare_prompts = {k.lower(): v for k, v in tejcare_prompts.items()}
keys = list(tejcare_prompts.keys())

# 🌱 Fallback Responses
fallback_lines = [
    "I'm here with you — no need to explain everything at once.",
    "Your feelings are valid, even if they’re hard to name.",
    "Let’s take this moment gently, together.",
    "You don’t have to be okay to be worthy of care.",
    "I’m listening — even to the silence between your words.",
    "You’re not alone in this. I’m here to sit with you.",
    "It’s okay to feel messy. You’re still lovable.",
    "Let’s breathe together. Inhale… exhale… again.",
    "You’re allowed to feel what you feel — no judgment here.",
    "Even when you don’t know what to say, I’m still here.",
    "You matter. Your presence matters. Your pain matters.",
    "Let’s hold space for your truth, whatever it looks like.",
    "You’re not broken — you’re becoming.",
    "You don’t have to fix anything right now. Just be.",
    "I’m proud of you for showing up here.",
    "Let’s take one soft step forward together.",
    "You’re allowed to rest. You’re allowed to feel.",
    "I see your effort, even when it’s invisible.",
    "You’re not too much. You’re just enough.",
    "Let’s sit in this moment — no rush, no pressure."
]

# 🧠 Semantic Matching Setup
model = SentenceTransformer('all-MiniLM-L6-v2')
key_embeddings = model.encode(keys, convert_to_tensor=True)

def get_semantic_reply(user_input):
    input_embedding = model.encode(user_input, convert_to_tensor=True)
    scores = util.cos_sim(input_embedding, key_embeddings)[0]
    best_score = scores.max().item()
    best_match_idx = scores.argmax().item()
    if best_score > 0.6:
        return tejcare_prompts[keys[best_match_idx]]
    else:
        return random.choice(fallback_lines)

# 📝 User Input
user_input = st.text_area("💬 Your message:", height=120, placeholder="e.g. I feel anxious...")

if st.button("Send"):
    msg = user_input.lower().strip()
    if msg:
        reply = get_semantic_reply(msg)
        st.divider()
        st.markdown("### 🌱 TejCare Response:")
        st.success(reply)
    else:
        st.warning("Please type something you're feeling — even a single word.")

# Footer
st.markdown("<hr><center><i>Built by Tejas · A chatbot that understands emotions, not just text 💙</i></center>", unsafe_allow_html=True)
