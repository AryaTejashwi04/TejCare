import streamlit as st

# Page config
st.set_page_config(page_title="TejCare Mental Health ChatBot", page_icon="🌿", layout="wide")

# Title and subtitle
st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: #4CAF50;'>🌿 TejCare - Mental Health ChatBot</h1>
        <p style='font-size: 18px;'>A gentle companion when you're struggling. Choose a feeling or say hello.</p>
    </div>
""", unsafe_allow_html=True)

st.divider()

# Initialize session state
if "response" not in st.session_state:
    st.session_state.response = ""

# Predefined prompts and responses
prompts_responses = {
    "Hi": "Hello! I'm TejCare, your emotional support companion. How are you feeling today?",
    "Hello": "Hey there! I'm here to listen and support you.",
    "Hey": "Hi! You can share anything with me — I'm here for you.",
    "I'm feeling anxious": "Anxiety can feel overwhelming. Try grounding yourself by focusing on your breath. You're safe, and this moment will pass.",
    "I'm sad": "Sadness is a valid emotion. Allow yourself to feel it without judgment. You're not alone, and healing begins with acknowledgment.",
    "I can't sleep": "Sleep struggles are tough. Try a calming routine—dim lights, soothing music, and deep breathing. Your body deserves rest.",
    "I feel lonely": "Loneliness can be heavy. Reach out to someone you trust, or even write your thoughts down. I'm here to listen.",
    "I'm overwhelmed": "When everything feels too much, pause. Break tasks into small steps. You don’t have to do it all at once.",
    "I need motivation": "Motivation ebbs and flows. Start with one small action—it often leads to momentum. You’ve got this.",
    "I'm stressed about exams": "Exam stress is real. Remember, your worth isn’t defined by scores. Prepare, breathe, and believe in yourself.",
    "I feel lost": "Feeling lost means you're searching. That’s brave. Let’s explore what matters to you, one step at a time.",
    "I miss someone": "Missing someone shows how deeply you care. Honor that feeling—it’s a reflection of love.",
    "I'm scared": "Fear is a natural response. You’re not weak for feeling it. Let’s talk about what’s causing it.",
    "I feel empty": "Emptiness can feel confusing. Try reconnecting with something that once brought you joy. Even small sparks matter.",
    "I want to cry": "Crying is healing. Let the tears flow—they carry emotion that needs release. I’m here with you.",
    "I feel hopeless": "Hopelessness is heavy. But even in darkness, a flicker of light exists. Let’s search for it together.",
    "I'm tired of everything": "Burnout is real. You deserve rest—not just sleep, but emotional space. Let’s slow down.",
    "I feel stuck": "Feeling stuck doesn’t mean failure. It means pause. Let’s find a new angle or path forward.",
    "I need a break": "Breaks aren’t weakness—they’re wisdom. Step back, breathe, and return when you’re ready.",
    "I feel guilty": "Guilt can teach us, but it shouldn’t punish us. Reflect, forgive, and grow. You’re human.",
    "I can't focus": "Focus fades when emotions rise. Try a short walk, hydration, or silence. Then return gently.",
    "I feel rejected": "Rejection hurts. But it doesn’t define your value. You are still worthy and loved.",
    "I'm angry": "Anger is valid. Let’s express it safely—write, talk, or move. Suppressing it only deepens the wound.",
    "I feel like a failure": "Failure is feedback, not identity. You’re learning, evolving, and still worthy.",
    "I need someone to talk to": "Talking helps. I’m here to listen. You don’t have to carry this alone.",
    "I feel numb": "Numbness is a shield. Let’s gently explore what’s beneath it, without pressure.",
    "I feel insecure": "Insecurity lies. You are enough, just as you are. Let’s challenge those thoughts together.",
    "I feel unloved": "You are loved—even if it’s hard to feel. Let’s reconnect with that truth.",
    "I feel like giving up": "Giving up feels tempting when pain is deep. But hold on—this moment isn’t the end.",
    "I feel broken": "Brokenness is part of being human. But you’re still whole in your essence.",
    "I feel worthless": "Worth isn’t earned—it’s inherent. You matter, even when you doubt it.",
    "I feel anxious in public": "Social anxiety is common. Try grounding techniques—touch, breath, or affirmations. You’re doing better than you think.",
    "I feel judged": "Judgment from others doesn’t define you. Your truth matters more than their opinion.",
    "I feel like nobody understands": "Feeling misunderstood is isolating. But your story deserves to be heard. I’m listening.",
    "I feel like I'm not enough": "You are enough. Not because of achievements—but because you exist, and that’s powerful.",
    "I feel like I'm drowning": "Drowning in emotion is exhausting. Let’s find a lifeline—one small act of self-care.",
    "I feel like I'm falling behind": "Comparison steals joy. Your pace is valid. Progress isn’t a race.",
    "I feel like I'm losing control": "Control slipping feels scary. Let’s focus on what you *can* influence—your breath, your choices.",
    "I feel like I'm disappointing people": "You’re doing your best. That’s enough. Others’ expectations aren’t your burden.",
    "I feel like I'm not heard": "Your voice matters. Let’s find ways to express it safely and clearly.",
    "I feel like I'm invisible": "You are seen. Even when the world feels distant, your presence matters.",
    "I feel like I'm always wrong": "Mistakes don’t define you. Growth does. Let’s learn, not punish.",
    "I feel like I'm too sensitive": "Sensitivity is strength. You feel deeply—and that’s a gift.",
    "I feel like I'm too emotional": "Emotions are valid. You’re not “too” anything. You’re beautifully human.",
    "I feel like I'm not strong": "Strength isn’t loud. It’s surviving quietly. You’re stronger than you know.",
    "I feel like I'm not smart": "Intelligence isn’t one-size-fits-all. You shine in your own way.",
    "I feel like I'm not good looking": "Beauty is diverse. You are radiant in ways that matter.",
    "I feel like I'm not talented": "Talent blooms in time. You’re growing—don’t rush the process.",
    "I feel like I'm not worthy": "Worth isn’t earned—it’s yours by birth. You are enough.",
    "I feel like I'm not wanted": "You belong. Even when it’s hard to feel, you matter deeply.",
    "I feel like I'm not loved": "Love surrounds you—even if quietly. Let’s reconnect with it.",
    "I feel like I'm not safe": "Safety starts with connection. You’re not alone. Let’s build calm together.",
    "I feel like I'm not okay": "Not being okay is okay. Let’s talk through it, gently."
}

# Display buttons in grid layout
st.markdown("### 💡 Choose a feeling or greeting:")
cols = st.columns(4)
keys = list(prompts_responses.keys())

for i in range(0, len(keys), 4):
    for j in range(4):
        if i + j < len(keys):
            with cols[j]:
                if st.button(keys[i + j]):
                    st.session_state.response = prompts_responses[keys[i + j]]

# Show response
if st.session_state.response:
    st.divider()
    st.markdown("### 🧠 TejCare Response:")
    st.success(st.session_state.response)