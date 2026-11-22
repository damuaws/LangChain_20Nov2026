import os

import pandas as pd
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


MODEL_NAME = "gpt-4o-mini"
JOB_ROLE = "Python Developer"

SAMPLE_APPLICATION = """Name: Jane Doe
Role: Python Developer
Experience: 6 Years
Skills: Python, Django, FastAPI, PostgreSQL, AWS"""


# ---------- API key ----------
def configure_api_key():
    """Read the key from Streamlit secrets (deployed app) or an environment variable (local)."""
    try:
        if "OPENAI_API_KEY" in st.secrets:
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass  # no secrets.toml locally; rely on the environment variable


# ---------- State ----------
class State(TypedDict):
    application: str
    experience_raw: str
    experience_level: str
    skill_raw: str
    skill_match: str
    decision: str
    response: str


# ---------- Output parsing (robust to verbose LLM replies) ----------
def parse_level(text: str) -> str:
    t = text.lower()
    if "senior" in t:
        return "Senior-level"
    if "mid" in t:
        return "Mid-level"
    if "entry" in t:
        return "Entry-level"
    return "Unknown"


def parse_match(text: str) -> str:
    t = text.lower()
    if "no match" in t:
        return "No Match"
    if "match" in t:
        return "Match"
    return "No Match"


# ---------- Graph ----------
@st.cache_resource(show_spinner=False)
def build_graph():
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

    experience_prompt = ChatPromptTemplate.from_template(
        "Based on the following job application, categorize the candidate as "
        "'Entry-level', 'Mid-level' or 'Senior-level'. "
        "Reply with only the category.\n\nApplication: {application}"
    )
    skill_prompt = ChatPromptTemplate.from_template(
        f"Based on the job application for a {JOB_ROLE}, assess the candidate's skillset. "
        "Reply with only 'Match' or 'No Match'.\n\nApplication: {application}"
    )

    def categorize_experience(state: State) -> dict:
        raw = (experience_prompt | llm).invoke({"application": state["application"]}).content
        return {"experience_raw": raw, "experience_level": parse_level(raw)}

    def assess_skillset(state: State) -> dict:
        raw = (skill_prompt | llm).invoke({"application": state["application"]}).content
        return {"skill_raw": raw, "skill_match": parse_match(raw)}

    def schedule_hr_interview(state: State) -> dict:
        return {
            "decision": "Shortlisted",
            "response": "Candidate has been shortlisted for an HR interview.",
        }

    def escalate_to_recruiter(state: State) -> dict:
        return {
            "decision": "Escalated",
            "response": "Candidate has senior-level experience but doesn't match job skills.",
        }

    def reject_application(state: State) -> dict:
        return {
            "decision": "Rejected",
            "response": "Candidate doesn't meet JD and has been rejected.",
        }

    def route_app(state: State) -> str:
        if state["skill_match"] == "Match":
            return "schedule_hr_interview"
        if state["experience_level"] == "Senior-level":
            return "escalate_to_recruiter"
        return "reject_application"

    graph = StateGraph(State)
    graph.add_node("categorize_experience", categorize_experience)
    graph.add_node("assess_skillset", assess_skillset)
    graph.add_node("schedule_hr_interview", schedule_hr_interview)
    graph.add_node("escalate_to_recruiter", escalate_to_recruiter)
    graph.add_node("reject_application", reject_application)

    graph.add_edge(START, "categorize_experience")
    graph.add_edge("categorize_experience", "assess_skillset")
    graph.add_conditional_edges(
        "assess_skillset",
        route_app,
        {
            "schedule_hr_interview": "schedule_hr_interview",
            "escalate_to_recruiter": "escalate_to_recruiter",
            "reject_application": "reject_application",
        },
    )
    graph.add_edge("schedule_hr_interview", END)
    graph.add_edge("escalate_to_recruiter", END)
    graph.add_edge("reject_application", END)

    return graph.compile()


def screen_candidate(application: str) -> dict:
    return build_graph().invoke({
        "application": application,
        "experience_raw": "",
        "experience_level": "",
        "skill_raw": "",
        "skill_match": "",
        "decision": "",
        "response": "",
    })


# ---------- UI ----------
def load_sample():
    st.session_state["application_text"] = SAMPLE_APPLICATION


def show_decision(decision: str, message: str):
    if decision == "Shortlisted":
        st.success(f"**{decision}**: {message}")
    elif decision == "Escalated":
        st.warning(f"**{decision}**: {message}")
    else:
        st.error(f"**{decision}**: {message}")


def main():
    st.set_page_config(page_title="HR Screening Agent", page_icon="🧑‍💼", layout="wide")
    configure_api_key()

    st.title("🧑‍💼 HR Screening Agent")
    st.caption(f"Screens candidates for the role of {JOB_ROLE} using an LLM-powered LangGraph workflow.")

    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "application_text" not in st.session_state:
        st.session_state["application_text"] = ""

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Candidate application")
        uploaded = st.file_uploader("Upload a .txt file (optional)", type=["txt"])
        if uploaded is not None:
            st.session_state["application_text"] = uploaded.read().decode("utf-8")

        application = st.text_area(
            "Or paste the application here",
            height=260,
            key="application_text",
        )

        b1, b2 = st.columns(2)
        b1.button("Load sample", on_click=load_sample)
        run = b2.button("Screen candidate", type="primary")

    with right:
        st.subheader("Result")
        if run:
            if not application.strip():
                st.warning("Please paste or upload an application first.")
            else:
                with st.spinner("Running screening workflow..."):
                    try:
                        result = screen_candidate(application)
                    except Exception as e:
                        st.error(f"Screening failed: {e}")
                        st.info("Check that OPENAI_API_KEY is set in Streamlit secrets or the environment.")
                        st.stop()

                c1, c2, c3 = st.columns(3)
                c1.metric("Experience level", result["experience_level"])
                c2.metric("Skill match", result["skill_match"])
                c3.metric("Decision", result["decision"])

                show_decision(result["decision"], result["response"])

                with st.expander("Raw LLM outputs"):
                    st.markdown("**Experience step:**")
                    st.write(result["experience_raw"])
                    st.markdown("**Skill step:**")
                    st.write(result["skill_raw"])

                st.session_state["history"].append({
                    "Candidate preview": application.strip()[:60].replace("\n", " ") + "...",
                    "Experience": result["experience_level"],
                    "Skill match": result["skill_match"],
                    "Decision": result["decision"],
                })
        else:
            st.info("Submit an application to see the screening decision.")

    if st.session_state["history"]:
        st.divider()
        st.subheader("Screening history (this session)")
        st.dataframe(pd.DataFrame(st.session_state["history"]))


if __name__ == "__main__":
    main()
