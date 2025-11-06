import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
from langchain.tools import DuckDuckGoSearchResults

# ---------- APP CONFIG ----------
st.set_page_config(page_title="AI Tools for Final-Year Projects", layout="wide")
st.title("🎓 Final-Year Project AI Tools")
st.sidebar.title("🧩 Features")
choice = st.sidebar.radio("Select a tool", ["📄 PDF Chatbot (Gemini)", "🌐 Web Plagiarism Checker"])

# ---------- GEMINI SETUP ----------
import os
genai.configure(api_key=os.getenv("AIzaSyCT-MKDxAOKKopCJPllQddrkVHzwswf15Q"))

# ---------- PDF CHATBOT (Gemini) ----------
if choice == "📄 PDF Chatbot (Gemini)":
    st.header("🤖 Chat with Your PDF (Gemini API)")
    pdf_file = st.file_uploader("Upload your project PDF", type=["pdf"])

    if pdf_file:
        reader = PdfReader(pdf_file)
        text = "".join(page.extract_text() or "" for page in reader.pages)

        st.success("✅ PDF uploaded successfully.")
        question = st.text_input("Ask a question about your PDF:")

        if question:
            with st.spinner("Analyzing PDF with Gemini..."):
                model = genai.GenerativeModel("gemini-1.5-flash")
                prompt = f"Answer this based only on the text below:\n\n{text}\n\nQuestion: {question}"
                response = model.generate_content(prompt)
                st.write("**Answer:**", response.text)
    else:
        st.info("Upload a PDF to start chatting.")

# ---------- WEB PLAGIARISM CHECKER ----------
elif choice == "🌐 Web Plagiarism Checker":
    st.header("🌍 Web-Based Plagiarism Checker")

    search_tool = DuckDuckGoSearchResults()
    model = SentenceTransformer("all-MiniLM-L6-v2")

    st.write("Enter a paragraph to check if similar content exists online.")
    user_text = st.text_area("Paste the paragraph here")

    if st.button("Check for Web Similarity"):
        if user_text.strip():
            with st.spinner("Searching the web..."):
                results = search_tool.run(user_text)

                emb1 = model.encode(user_text, convert_to_tensor=True)
                emb2 = model.encode(results, convert_to_tensor=True)
                score = util.cos_sim(emb1, emb2).item()

                st.subheader("Similarity Result")
                st.metric("Similarity Score", f"{score:.2f}")

                if score > 0.8:
                    st.error("⚠️ High similarity — likely plagiarized.")
                elif score > 0.5:
                    st.warning("⚠️ Moderate similarity — review recommended.")
                else:
                    st.success("✅ Low similarity — appears original.")

                st.write("**Search snippet used:**")
                st.code(results[:1000])
        else:
            st.warning("Please enter text before checking.")
