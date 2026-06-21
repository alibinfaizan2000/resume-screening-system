import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import os
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_BASE = f"{BACKEND_URL}/api/v1"

st.set_page_config(
    page_title="RecruitAI — Resume Screening System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main { background-color: #0f1117; }
[data-testid="stSidebar"] { background-color: #161b27; border-right: 1px solid #2d3748; }

.metric-card {
    background: linear-gradient(135deg, #1a2035 0%, #1e2d40 100%);
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 20px;
    margin: 8px 0;
}
.candidate-card {
    background: #161b27;
    border: 1px solid #2d3748;
    border-left: 4px solid #4f46e5;
    border-radius: 10px;
    padding: 20px;
    margin: 12px 0;
}
.candidate-card.top-pick { border-left-color: #10b981; }
.candidate-card.no-hire { border-left-color: #ef4444; }
.source-chip {
    display: inline-block;
    background: #1e2d40;
    border: 1px solid #4f46e5;
    color: #818cf8;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 12px;
    margin: 3px;
    font-family: 'JetBrains Mono', monospace;
}
.skill-tag {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 12px;
    margin: 2px;
    font-weight: 500;
}
.skill-match { background: #064e3b; color: #6ee7b7; border: 1px solid #10b981; }
.skill-miss { background: #450a0a; color: #fca5a5; border: 1px solid #ef4444; }
.stButton button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 24px;
    transition: all 0.2s;
}
.stButton button:hover { opacity: 0.9; transform: translateY(-1px); }
h1, h2, h3 { color: #f1f5f9 !important; }
.stTextArea textarea, .stTextInput input {
    background: #1a2035 !important;
    border: 1px solid #2d3748 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
.chat-message {
    padding: 14px 18px;
    border-radius: 10px;
    margin: 8px 0;
}
.chat-user { background: #1e2d40; border-left: 3px solid #4f46e5; }
.chat-assistant { background: #162032; border-left: 3px solid #10b981; }
</style>
""", unsafe_allow_html=True)


def sidebar():
    with st.sidebar:
        st.markdown("## 🎯 RecruitAI")
        st.markdown("*AI-Powered Resume Screening*")
        st.divider()

        page = st.radio(
            "Navigation",
            ["📤 Upload Resumes", "🏆 Candidate Ranking", "💬 Resume Q&A", "🔍 Search"],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("**Stack**")
        st.markdown("""
- 🧠 Gemini 1.5 Pro
- 📦 Pinecone Vector DB
- 🔗 LangGraph Workflow
- 🔍 Hybrid Search
- 📊 LangSmith Tracing
        """)

        try:
            r = requests.get(f"{BACKEND_URL}/health", timeout=3)
            if r.status_code == 200:
                st.success("✅ Backend Online")
            else:
                st.error("❌ Backend Error")
        except Exception:
            st.error("❌ Backend Offline")

    return page


def upload_page():
    st.markdown("# 📤 Upload Resumes")
    st.markdown("Upload candidate PDF resumes to index them into the vector database.")

    uploaded_files = st.file_uploader(
        "Drop PDF resumes here",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF resumes",
    )

    if uploaded_files:
        st.info(f"📄 {len(uploaded_files)} file(s) selected")

        if st.button("🚀 Index Resumes", use_container_width=True):
            progress = st.progress(0)
            results_placeholder = st.empty()
            results = []

            for i, f in enumerate(uploaded_files):
                with st.spinner(f"Indexing {f.name}..."):
                    try:
                        r = requests.post(
                            f"{API_BASE}/resumes/upload",
                            files=[("files", (f.name, f.getvalue(), "application/pdf"))],
                            timeout=120,
                        )
                        if r.status_code == 200:
                            data = r.json()
                            results.extend(data)
                        else:
                            results.append({
                                "filename": f.name,
                                "status": "error",
                                "message": r.text,
                                "chunks_indexed": 0,
                            })
                    except Exception as e:
                        results.append({
                            "filename": f.name,
                            "status": "error",
                            "message": str(e),
                            "chunks_indexed": 0,
                        })
                progress.progress((i + 1) / len(uploaded_files))

            st.markdown("### Results")
            for res in results:
                if res["status"] == "success":
                    st.success(f"✅ **{res['filename']}** — {res['chunks_indexed']} chunks indexed")
                else:
                    st.error(f"❌ **{res['filename']}** — {res['message']}")

    st.divider()
    try:
        r = requests.get(f"{API_BASE}/resumes/index-status", timeout=10)
        if r.status_code == 200:
            data = r.json()
            st.metric("Total Chunks in Vector DB", data.get("total_chunks", 0))
    except Exception:
        pass


def ranking_page():
    st.markdown("# 🏆 Candidate Ranking")

    col1, col2 = st.columns([2, 1])
    with col1:
        jd = st.text_area(
            "Job Description",
            height=200,
            placeholder="Paste the full job description here...",
        )
    with col2:
        top_k = st.slider("Max Candidates", 1, 10, 5)
        st.markdown("#### Sample JDs")
        if st.button("ML Engineer"):
            st.session_state["sample_jd"] = "We are looking for a Senior ML Engineer with expertise in Python, PyTorch, TensorFlow, scikit-learn, and MLOps. Experience with LLMs, RAG systems, vector databases, and production model deployment is required. FastAPI, Docker, and cloud platforms (AWS/GCP) are a plus."
        if st.button("Data Scientist"):
            st.session_state["sample_jd"] = "Data Scientist role requiring Python, R, statistical modeling, machine learning, SQL, and data visualization. Experience with A/B testing, feature engineering, and deploying models to production preferred."

    if "sample_jd" in st.session_state:
        jd = st.session_state["sample_jd"]

    if st.button("🎯 Match Candidates", use_container_width=True, disabled=not jd):
        with st.spinner("Running hybrid retrieval, reranking, and AI analysis..."):
            try:
                r = requests.post(
                    f"{API_BASE}/matching/candidates",
                    json={"job_description": jd, "top_k": top_k},
                    timeout=180,
                )
                if r.status_code == 200:
                    data = r.json()
                    st.session_state["ranking_data"] = data
                    st.success(f"Analyzed {data['total_candidates_analyzed']} candidates")
                else:
                    st.error(f"Error: {r.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

    if "ranking_data" in st.session_state:
        data = st.session_state["ranking_data"]
        candidates = data.get("candidates", [])

        if not candidates:
            st.warning("No candidates found. Upload resumes first.")
            return

        st.markdown("### 📊 Match Score Overview")
        fig = go.Figure(go.Bar(
            x=[c["candidate_name"] for c in candidates],
            y=[c["match_percentage"] for c in candidates],
            marker_color=[
                "#10b981" if c["match_percentage"] >= 70 else
                "#f59e0b" if c["match_percentage"] >= 50 else "#ef4444"
                for c in candidates
            ],
            text=[f"{c['match_percentage']:.0f}%" for c in candidates],
            textposition="outside",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            yaxis=dict(range=[0, 110], gridcolor="#2d3748"),
            xaxis=dict(gridcolor="#2d3748"),
            margin=dict(t=20, b=20),
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🎯 Candidate Analysis")
        for i, c in enumerate(candidates):
            card_class = "top-pick" if i == 0 else ("no-hire" if c["match_percentage"] < 40 else "")
            badge = "🥇 TOP PICK" if i == 0 else f"#{i+1}"

            with st.expander(f"{badge} {c['candidate_name']} — {c['match_percentage']:.0f}% match | {c['recommendation']}", expanded=(i == 0)):
                col1, col2, col3 = st.columns(3)
                col1.metric("Match", f"{c['match_percentage']:.0f}%")
                col2.metric("Recommendation", c["recommendation"])
                col3.metric("File", c["filename"])

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**✅ Matching Skills**")
                    skills_html = " ".join([f'<span class="skill-tag skill-match">{s}</span>' for s in c["matching_skills"]])
                    st.markdown(skills_html or "*None identified*", unsafe_allow_html=True)

                    st.markdown("**💪 Strengths**")
                    for s in c["strengths"]:
                        st.markdown(f"• {s}")

                with col_b:
                    st.markdown("**❌ Missing Skills**")
                    skills_html = " ".join([f'<span class="skill-tag skill-miss">{s}</span>' for s in c["missing_skills"]])
                    st.markdown(skills_html or "*None identified*", unsafe_allow_html=True)

                    st.markdown("**⚠️ Weaknesses**")
                    for w in c["weaknesses"]:
                        st.markdown(f"• {w}")

                st.markdown("**📝 Reasoning**")
                st.info(c["reasoning"])

                if c["retrieved_chunks"]:
                    st.markdown("**📄 Evidence Chunks**")
                    for chunk in c["retrieved_chunks"][:2]:
                        with st.container():
                            st.markdown(f'<span class="source-chip">Score: {chunk.get("score", 0):.3f}</span>', unsafe_allow_html=True)
                            st.code(chunk["text"][:300], language=None)


def qa_page():
    st.markdown("# 💬 Resume Q&A")
    st.markdown("Ask natural language questions about the indexed candidates.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    sample_questions = [
        "Who is the best candidate for an ML Engineer role?",
        "Which candidates have RAG or LLM experience?",
        "Which candidate has the strongest Python background?",
        "Compare the top two candidates.",
        "Who has production deployment experience?",
    ]

    st.markdown("**Quick Questions:**")
    cols = st.columns(len(sample_questions))
    for i, q in enumerate(sample_questions):
        if cols[i].button(q[:30] + "...", key=f"quick_{i}"):
            st.session_state["quick_question"] = q

    jd_context = st.text_input(
        "Job Description Context (optional)",
        placeholder="Add a job description to make answers more relevant...",
    )

    for msg in st.session_state.messages:
        css_class = "chat-user" if msg["role"] == "user" else "chat-assistant"
        icon = "👤" if msg["role"] == "user" else "🤖"
        st.markdown(
            f'<div class="chat-message {css_class}"><strong>{icon}</strong> {msg["content"]}</div>',
            unsafe_allow_html=True,
        )
        if msg["role"] == "assistant" and msg.get("sources"):
            score = msg.get("faithfulness_score", 0.0)
            passed = msg.get("faithfulness_passed", False)
            badge = "✅ Grounded" if passed else "⚠️ Partially Grounded"
            color = "#10b981" if passed else "#f59e0b"
            st.markdown(
                f'<span style="background:{color}22;border:1px solid {color};color:{color};'
                f'padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600;">'
                f'{badge} — Faithfulness: {score:.2f}</span>',
                unsafe_allow_html=True,
            )
            if msg.get("unsupported_claims"):
                with st.expander("⚠️ Unsupported Claims Detected"):
                    for claim in msg["unsupported_claims"]:
                        st.warning(claim)
            with st.expander("📚 Retrieved Sources"):
                for src in msg["sources"]:
                    st.markdown(f'<span class="source-chip">{src["filename"]}</span>', unsafe_allow_html=True)
                    st.code(src["chunk_text"][:300], language=None)
                    if src.get("relevance_score"):
                        st.caption(f"Relevance: {src['relevance_score']:.4f}")

    question = st.chat_input("Ask about candidates...")
    if "quick_question" in st.session_state:
        question = st.session_state.pop("quick_question")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("Searching resumes and generating answer..."):
            try:
                r = requests.post(
                    f"{API_BASE}/qa/ask",
                    json={"question": question, "job_description": jd_context or None},
                    timeout=120,
                )
                if r.status_code == 200:
                    data = r.json()
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data.get("sources", []),
                        "citations": data.get("citations", []),
                        "faithfulness_score": data.get("faithfulness_score", 0.0),
                        "faithfulness_passed": data.get("faithfulness_passed", False),
                        "unsupported_claims": data.get("unsupported_claims", []),
                    })
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error: {r.text}",
                        "sources": [],
                        "citations": [],
                    })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Connection error: {e}",
                    "sources": [],
                    "citations": [],
                })
        st.rerun()

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()


def search_page():
    st.markdown("# 🔍 Semantic Search")
    st.markdown("Search across all indexed resumes using hybrid retrieval.")

    query = st.text_input("Search Query", placeholder="e.g. FastAPI microservices Docker Kubernetes")
    top_k = st.slider("Results", 1, 15, 5)

    if st.button("🔍 Search", disabled=not query):
        with st.spinner("Running hybrid search and reranking..."):
            try:
                r = requests.post(
                    f"{API_BASE}/qa/search",
                    json={"query": query, "top_k": top_k},
                    timeout=60,
                )
                if r.status_code == 200:
                    data = r.json()
                    st.success(f"Found {data['total_results']} relevant chunks")

                    for i, res in enumerate(data["results"]):
                        with st.expander(f"Result {i+1} — {res['filename']}", expanded=(i == 0)):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f'<span class="source-chip">{res["chunk_id"]}</span>', unsafe_allow_html=True)
                            with col2:
                                if res.get("relevance_score"):
                                    st.metric("Score", f"{res['relevance_score']:.4f}")
                            st.markdown(res["chunk_text"])
                else:
                    st.error(r.text)
            except Exception as e:
                st.error(str(e))


def main():
    page = sidebar()

    if page == "📤 Upload Resumes":
        upload_page()
    elif page == "🏆 Candidate Ranking":
        ranking_page()
    elif page == "💬 Resume Q&A":
        qa_page()
    elif page == "🔍 Search":
        search_page()


if __name__ == "__main__":
    main()
