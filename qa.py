from dataclasses import dataclass
from typing import Optional

import streamlit as st
from transformers import pipeline

MODEL_ID = "deepset/roberta-base-squad2"


def get_pipeline():
    """
    Get the pipeline used to answer questions
    :return: the pipeline
    """
    if "pipeline" not in st.session_state:
        nlp = pipeline("question-answering", model=MODEL_ID, tokenizer=MODEL_ID)
        st.session_state.pipeline = nlp

    return st.session_state.pipeline


# noinspection PyUnresolvedReferences
def get_answer(question: str, context: str) -> str:
    """
    Get the answer to a question given a context
    :param question: the question
    :param context: the context
    :return: the answer
    """
    qa_input = {
        "question": question,
        "context": context
    }

    return get_pipeline()(qa_input)["answer"]


def main():
    st.set_page_config(
        page_title="Question and answers",
        page_icon="🤔",
        layout="wide"
    )

    st.title("Question and answers")
    st.text("This is an example of a question and answer system using a LLM model.")

    columns = st.columns(2)

    with columns[0]:
        st.header("Context")
        context = st.text_area("Enter the context here")

    with columns[1]:
        st.header("Question")
        question = st.text_input("Enter the question here")

    if st.button("Answer"):
        answer = get_answer(question, context)
        st.success(f"The answer is: {answer}")


if __name__ == '__main__':
    main()
