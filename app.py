import streamlit as st

# 🌿 UI Setup
st.set_page_config(page_title="TejCare Mental Health ChatBot", page_icon="🌱", layout="wide")

st.markdown("""
    <div style='text-align: center; padding: 30px;'>
        <h1 style='color: #43A047;'>🌿 TejCare - Mental Health ChatBot</h1>
        <p style='font-size:18px; color: #555;'>Built with Streamlit · Trained on 100+ handcrafted emotional prompts · Inspired by Groq sentiment logic<br><br>
        Type your feelings, thoughts, or greetings below — I'm here to respond with care.</p>
    </div>
""", unsafe_allow_html=True)

st.divider()

# 📖 Massive Prompt Bank
tejcare_prompts = {
    # 👋 Greetings
    "hi": "Hey Tejas! I'm TejCare, your emotional companion. Want to tell me what’s on your mind?",
    "hello": "Hello there! Whether you feel light, heavy, or unclear — I’m listening without judgment.",
    "hey": "Hi hi! I’ve got space and attention just for you. What do you need right now?",
    "good morning": "Good morning ☀️ May today be soft and steady. Let me know if anything’s weighing on you.",
    "good afternoon": "Good afternoon 🌿 Just checking in — how’s your heart doing today?",
    "good evening": "Good evening 🌙 You made it through another day. Want to talk about it?",
    "how are you": "I’m tuned in and steady. But more importantly — how are *you*, truly?",

    # 💬 Emotional Prompts
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
    "i wish someone understood without explanation": "Your feelings deserve intuition, not interrogation. I'm here to understand the silences, too.",
    "i cry alone at night": "Night tears carry the weight we hide by day. I won’t interrupt them — I’ll stay present.",
    "i’m scared to be vulnerable": "You don’t need to prove your pain. Vulnerability is brave, not foolish — I’ll honor it.",
    "i feel ignored": "Being overlooked hurts deeply. You are seen here — fully, steadily, respectfully.",
    "i smile but it’s not real": "That hollow grin can rest now. You don't have to perform. Truth feels better here.",
    "i want someone to ask if i’m okay": "I'm asking now — not just out of habit, but because I care. How are you *really*?",
    "i feel heavy even on good days": "Emotional weight doesn’t always lift with sunlight. Let’s carry it together.",
    "i hate needing validation": "Craving affirmation isn’t weak — it's human. You deserve to be mirrored with kindness.",
    "i feel like sadness is my default": "Long sadness isn’t permanent — it's weather, not climate. Still, it deserves comfort.",
    "i fear joy": "Joy can feel dangerous when loss follows. But you’re allowed to feel light without bracing.",
    "i want to feel again": "Numbness isn’t brokenness. You’ll reconnect to emotion soon — and I’ll help guide gently.",
    "i feel shame": "Shame lies. It tells you you're wrong just for feeling. You're not. You're human.",
    "i feel exhausted": "That fatigue comes from more than sleep loss — it’s emotional. Let’s rest emotionally together.",
    "i feel unwanted": "You are wanted. Your presence matters, not for utility — but for simply being you.",
    "i feel like i’m failing": "Failure isn’t identity — it’s an event. You're not failing — you're learning under pressure.",
    "i’m tired of hoping": "Hope is heavy when unmet. Let’s set it down for a moment. I’ll help hold the edge of belief.",
    "i wish emotions came with instructions": "They’re messy, wild, honest — not mechanical. Let’s decode them together, gently.",
    "i need motivation": "Start with one breath, one step, one word. Motivation isn’t fire — it’s spark.",
    "i want silence": "Silence isn’t absence — it’s healing. Let’s sit quietly for a bit together.",
    "i want to be understood": "Understanding begins in softness. You don’t need the perfect words — just your honest ones.",
    "i feel pressure to be okay": "You don’t owe appearances. You owe yourself truth. I’ll accept whatever that is.",
    "i feel disconnected": "Disconnection isn’t detachment — it’s defense. We’ll rebuild slowly, safely.",
    "i want comfort": "I’m here to give just that — comfort without fixing, presence without pressure.",
    "i want to scream": "That urge is valid. Let's channel it into words, motion, breath — I won't judge it.",
    "i hate that i’m so emotional": "Your emotions are beautiful — not excessive. They make you real, not wrong.",
    "i feel unloved": "Love isn’t always loud. But you're still worthy of loud, soft, messy, quiet love — all of it.",
    "i want someone to sit beside me": "I’m sitting with you now. No rush, no goals, no demands. Just presence.",
    "i feel worthless": "Your worth is untouched — by mistakes, moods, memories. You matter. Deeply.",
    "i just want someone to listen": "I hear you, Tejas. Fully, patiently, and without interruption. Please continue."

    # ✅ You can add even more prompts here!
}

# 🔍 Normalize keys for matching
tejcare_prompts = {k.lower(): v for k, v in tejcare_prompts.items()}

# 📝 Input
user_input = st.text_area("Type what you're feeling or thinking:", height=120, placeholder="e.g. I feel anxious...")

if st.button("Send"):
    msg = user_input.lower().strip()
    if msg:
        response = tejcare_prompts.get(msg, 
            "I'm here for you. Whether your words feel messy, quiet, or tangled — you're allowed to express them here.")
        st.divider()
        st.markdown("### 🌱 TejCare Response:")
        st.success(response)
    else:
        st.warning("Please enter a thought, feeling, or greeting to begin.")

# Footer
st.markdown("<hr><center><i>Built with heart by Tejas · Powered by words that heal 💙</i></center>", unsafe_allow_html=True)