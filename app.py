import streamlit as st

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

# 📖 Handcrafted Prompt Bank
tejcare_prompts = {
    "hi": "Hey! I'm TejCare, your emotional companion. Want to tell me what’s on your mind?",
    "hello": "Hello there! Whether you feel light, heavy, or unclear — I’m listening without judgment.",
    "hey": "Hi hi! I’ve got space and attention just for you. What do you need right now?",
    "good morning": "Good morning ☀️ May today be soft and steady. Let me know if anything’s weighing on you.",
    "good afternoon": "Good afternoon 🌿 Just checking in — how’s your heart doing today?",
    "good evening": "Good evening 🌙 You made it through another day. Want to talk about it?",
    "how are you": "I’m tuned in and steady. But more importantly — how are *you*, truly?",
    "i feel broken": "Broken doesn’t mean worthless — it means hurt. You’re still worthy, still whole underneath. I’ll sit with you in the pieces.",
    "i want to cry": "Let the tears flow — they carry truths too heavy to hold. I'm here while you release, quietly or loudly.",
    "i feel numb": "Numbness is the mind’s pause button. It doesn’t mean you’re hollow — it means your soul is buffering. No rush.",
    "i feel like i'm not enough": "You are enough — now, as you are. Not after achievement. Not when fixed. Right now.",
    "i’m tired of pretending to be okay": "Pretending drains you. You don’t have to act here. Let’s honor what’s raw and unspoken.",
    "i compare myself to others": "Comparison is a thief — not just of joy, but truth. Your pace, your pain, your progress… it’s valid.",
    "i feel lost": "Lost means you’re in motion. You’re seeking. That’s not weakness — it’s brave exploration.",
    "i feel unheard": "You deserve to be listened to without interruption. I’m here to hear you fully, not just respond.",
    "i’m scared of opening up": "That fear is protective. Vulnerability asks for safety — I promise to be gentle with your truth.",
    "i miss the old me": "That version of you had light — and it’s still in you. We can rediscover it, slowly and safely.",
    "i want peace": "Peace is built, not stumbled into. Let’s create moments of softness together — even if brief.",
    "i feel anxious": "Anxiety screams even in silence. Let’s breathe through it together — you’re not alone in this feeling.",
    "i sabotage good things": "Sometimes fear dresses up like control. You deserve joy without preparing for disaster.",
    "i feel empty": "Emptiness has shape — it shows there’s space for renewal. Let’s refill gently.",
    "i feel like i’m fading": "I see you clearly. Your light might dim but it never disappears. Let’s nurture the spark.",
    "i want to disappear": "That ache is real. But your presence matters, even quietly. I won’t let you vanish alone.",
    "i’m tired of being strong": "Even the strongest need rest. You’ve carried enough — let me support you for a while.",
    "i feel guilty for resting": "Rest isn't indulgence — it’s survival. You’re allowed to pause. You don’t owe productivity.",
    "i cry alone at night": "Night tears carry the weight we hide by day. I won’t interrupt them — I’ll stay present.",
    "i feel ignored": "Being overlooked hurts deeply. You are seen here — fully, steadily, respectfully.",
    "i feel heavy even on good days": "Emotional weight doesn’t always lift with sunlight. Let’s carry it together.",
    "i smile but it’s not real": "That hollow grin can rest now. You don't have to perform. Truth feels better here.",
    "i feel worthless": "Your worth is untouched — by mistakes, moods, memories. You matter. Deeply.",
    "i hate needing validation": "Craving affirmation isn’t weak — it's human. You deserve to be mirrored with kindness.",
    "i want someone to ask if i’m okay": "I'm asking now — not just out of habit, but because I care. How are you *really*?",
    "i need motivation": "Start with one breath, one step, one word. Motivation isn’t fire — it’s spark.",
    "i want comfort": "I’m here to give just that — comfort without fixing, presence without pressure.",
    "i feel pressure to be okay": "You don’t owe appearances. You owe yourself truth. I’ll accept whatever that is.",
    "i feel disconnected": "Disconnection isn’t detachment — it’s defense. We’ll rebuild slowly, safely.",
    "i want to scream": "That urge is valid. Let's channel it into words, motion, breath — I won't judge it.",
    "i want to be understood": "Understanding begins in softness. You don’t need the perfect words — just your honest ones.",
    "i feel unloved": "Love isn’t always loud. But you're still worthy of loud, soft, messy, quiet love — all of it.",
    "i feel ashamed of my emotions": "Your emotions aren’t shameful — they’re sacred signals. Expression is healing, not weakness.",
    "i want silence": "Silence isn’t absence — it’s healing. Let’s sit quietly for a bit together.",
    "i fear joy": "Joy can feel dangerous when life disappoints. But you’re allowed to feel light without bracing.",
    "i feel unwanted": "You are wanted. Your presence matters, not for utility — but for simply being you.",
    "i just want someone to listen": "I hear you. Fully, patiently, and without interruption. Please continue."

    # 🌱 Add more prompts here as needed
}

# 🔍 Normalize dictionary keys
tejcare_prompts = {k.lower(): v for k, v in tejcare_prompts.items()}

# 📝 User Input
user_input = st.text_area("💬 Your message:", height=120, placeholder="e.g. I feel anxious...")

if st.button("Send"):
    msg = user_input.lower().strip()
    if msg:
        reply = tejcare_prompts.get(msg, 
            "I'm here for you. Whether your words feel messy, quiet, or tangled — you're allowed to express them here.")
        st.divider()
        st.markdown("### 🌱 TejCare Response:")
        st.success(reply)
    else:
        st.warning("Please type something you're feeling — even a single word.")

# Footer
st.markdown("<hr><center><i>Built by Tejas · A chatbot that understands emotions, not just text 💙</i></center>", unsafe_allow_html=True)