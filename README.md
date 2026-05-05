🌿 TejCare
TejCare is a sophisticated Retrieval-Augmented Generation (RAG) platform designed to provide empathetic, research-grounded mental health support. By utilizing a dual-index FAISS architecture, the system balances sub-millisecond response latency with deep clinical accuracy.

🚀 Key Architectural Features
1. Dual-Index Pipeline
The core engine operates on a tiered retrieval strategy to optimize both speed and depth:

Index1 (L1 Semantic Cache): A high-speed FAISS vector store that caches previous successful Query-Response pairs. If a new query is >92% semantically similar to a cached entry, the system serves the response instantly, bypassing LLM costs and latency.

Index2 (L2 Clinical RAG): A vast repository of vectorized clinical research blocks inspired by WHO and Lancet mental health protocols. This ensures that every response is grounded in established medical frameworks.

2. Hybrid Sentiment Ensemble
TejCare doesn't just read text; it analyzes emotional subtext. The system runs a local ensemble of VADER and TextBlob sentiment engines to calculate a composite polarity score. This metadata is then injected into the LLM prompt to ensure the AI's tone perfectly mirrors the user's emotional state.

3. Recursive Safety Guardrail (Re-Ranker)
Security is paramount in mental health tech. Every generated response undergoes a Recursive Density Audit:

The system matches the output against a 90+ word safety lexicon.

If the "Harmful Token Density" exceeds 50%, the system automatically rejects the output and triggers a recursive re-generation until a safe, supportive response is produced.

4. Conversational Buffer Window Memory
Unlike standard chatbots that lose track of the conversation, TejCare maintains a Rolling Buffer Window of 50 turns. This allows for deep contextual continuity, making the interaction feel like a continuous journey rather than isolated messages.

🛠️ Tech Stack
LLM: Google Gemini 1.5 Flash

Vector Database: FAISS (Facebook AI Similarity Search)

Embeddings: Sentence-Transformers (all-MiniLM-L6-v2)

Frontend: Streamlit (Glassmorphism UI)

NLP: VADER Sentiment & TextBlob

📦 Installation & Setup
Clone the repository:

Bash
git clone https://github.com/your-username/tejcare.git
Install Dependencies:

Bash
pip install -r requirements.txt
Configure Secrets:
Create a .streamlit/secrets.toml file and add your API key:

Ini, TOML
   GEMINI_KEY = "YOUR_GOOGLE_AI_STUDIO_KEY"
Run the App:

Bash
streamlit run app.py
👨‍💻 Developer
Tejashwi Arya
NITK Surathkal | Electrical & Electronics Engineering
