import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import pdfplumber


# Function to extract text from PDF
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text.strip()


# Function for text summarization
def summarize_text(text, word_limit=100):
    summarizer = pipeline("summarization")
    try:
        return summarizer(text, max_length=word_limit, min_length=30, do_sample=False)[0]["summary_text"]
    except Exception:
        return "Summarization failed. The text might be too long or unsuitable for summarization."


# Function to generate questions based on text
def generate_questions(text, num_questions=5):
    sentences = text.split('.')
    questions = []
    for sentence in sentences[:num_questions]:
        words = sentence.split()
        if len(words) > 3:
            questions.append(f"What is the importance of: {' '.join(words[:3])}?")
    return questions


# Function to provide feedback using cosine similarity
def provide_feedback(response, reference_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([reference_text, response])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return f"Your response similarity with the reference text is {similarity * 100:.2f}%."


# Main Streamlit app
def main():
    st.title("Interactive PDF Learning and Assessment Platform 📘")

    menu = ["Upload & Summarize", "Generate Quiz", "Feedback System", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Upload & Summarize":
        st.subheader("Upload a PDF and Summarize 📂")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        word_limit = st.slider("Select Summary Word Limit", 10, 500, 100)
        if uploaded_file:
            try:
                text = extract_text_from_pdf(uploaded_file)
                if text.strip():
                    st.success("PDF uploaded and processed successfully!")
                    with st.expander("Extracted Text"):
                        st.text_area("Extracted Text", value=text, height=200)
                    summary = summarize_text(text, word_limit=word_limit)
                    st.subheader("Summary")
                    st.write(summary)
                else:
                    st.error("The PDF contains no readable text.")
            except Exception as e:
                st.error(f"Error processing PDF: {e}")

    elif choice == "Generate Quiz":
        st.subheader("Generate Quiz from Text 📝")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        num_questions = st.slider("Select Number of Questions", 1, 20, 5)
        if uploaded_file:
            try:
                text = extract_text_from_pdf(uploaded_file)
                if text.strip():
                    questions = generate_questions(text, num_questions=num_questions)
                    st.subheader("Generated Questions")
                    for i, q in enumerate(questions):
                        st.write(f"{i + 1}. {q}")
                else:
                    st.error("The PDF contains no readable text.")
            except Exception as e:
                st.error(f"Error processing PDF: {e}")

    elif choice == "Feedback System":
        st.subheader("Provide Feedback Based on a Question from the PDF 📄")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        if uploaded_file:
            try:
                text = extract_text_from_pdf(uploaded_file)
                if text.strip():
                    st.success("PDF uploaded and processed successfully!")
                    questions = generate_questions(text, num_questions=1)
                    if questions:
                        st.write("Question:")
                        question = questions[0]
                        st.write(f"**{question}**")

                        user_response = st.text_area("Your Response", height=150)
                        reference_text = text[:300]  # Take first 300 characters as a reference
                        if st.button("Get Feedback"):
                            if user_response.strip():
                                feedback = provide_feedback(user_response, reference_text)
                                st.write(feedback)
                            else:
                                st.warning("Please enter your response.")
                    else:
                        st.error("No suitable questions could be generated from the text.")
                else:
                    st.error("The PDF contains no readable text.")
            except Exception as e:
                st.error(f"Error processing PDF: {e}")

    elif choice == "About":
        st.subheader("About This App")
        st.info(
            "This app provides tools for summarizing PDFs, generating quizzes, "
            "and assessing responses using rule-based and NLP-based text analysis."
        )


if __name__ == "__main__":
    main()
