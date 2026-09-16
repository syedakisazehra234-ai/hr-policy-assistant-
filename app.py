import os
import tempfile

import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from groq import Groq


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="👩‍💼",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
        .main {
            padding-top: 1rem;
        }

        .title {
            font-size: 2.5rem;
            font-weight: 700;
        }

        .subtitle {
            font-size: 1.1rem;
            color: #666;
            margin-bottom: 2rem;
        }

        .source-box {
            background-color: #f5f5f5;
            padding: 12px;
            border-radius: 8px;
            margin-top: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="title">👩‍💼 HR Policy Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload an HR Policy PDF and ask questions about its contents.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# --------------------------------------------------
# GROQ CLIENT
# --------------------------------------------------

def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        st.error(
            "GROQ_API_KEY is not configured. "
            "Add it under Streamlit Cloud → Settings → Secrets."
        )
        st.stop()

    return Groq(api_key=api_key)


# --------------------------------------------------
# PDF TEXT EXTRACTION
# --------------------------------------------------

def extract_pdf_text(uploaded_file):
    """
    Extract text from uploaded PDF using PyMuPDF.
    """

    pdf_bytes = uploaded_file.getvalue()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(pdf_bytes)
        temp_path = temp_file.name

    try:
        document = fitz.open(temp_path)

        pages = []

        for page_number, page in enumerate(document):
            text = page.get_text("text")

            if text.strip():
                pages.append(
                    {
                        "page": page_number + 1,
                        "text": text
                    }
                )

        document.close()

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return pages


# --------------------------------------------------
# TEXT CHUNKING
# --------------------------------------------------

def create_chunks(pages, chunk_size=1000, overlap=150):
    """
    Split extracted PDF text into overlapping chunks.
    """

    chunks = []

    for page_data in pages:

        page_number = page_data["page"]
        text = page_data["text"]

        # Normalize whitespace
        text = " ".join(text.split())

        if not text:
            continue

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append(
                    {
                        "text": chunk_text.strip(),
                        "page": page_number
                    }
                )

            if end >= len(text):
                break

            start = end - overlap

    return chunks


# --------------------------------------------------
# CREATE FAISS INDEX
# --------------------------------------------------

def create_faiss_index(chunks):
    """
    Create Sentence Transformer embeddings
    and store them in a FAISS index.
    """

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embeddings = embeddings.astype("float32")

    # Normalize embeddings so inner product
    # behaves like cosine similarity.
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# --------------------------------------------------
# RETRIEVE RELEVANT CHUNKS
# --------------------------------------------------

def retrieve_chunks(question, index, chunks, top_k=5):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    question_embedding = question_embedding.astype("float32")

    faiss.normalize_L2(question_embedding)

    scores, indices = index.search(
        question_embedding,
        min(top_k, len(chunks))
    )

    retrieved = []

    for score, index_position in zip(scores[0], indices[0]):

        if index_position == -1:
            continue

        retrieved.append(
            {
                "text": chunks[index_position]["text"],
                "page": chunks[index_position]["page"],
                "score": float(score)
            }
        )

    return retrieved


# --------------------------------------------------
# GENERATE ANSWER USING GROQ
# --------------------------------------------------

def generate_answer(question, retrieved_chunks):

    client = get_groq_client()

    context_parts = []

    for item in retrieved_chunks:

        context_parts.append(
            f"[Page {item['page']}]\n"
            f"{item['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are an HR Policy Assistant.

Your job is to answer questions using ONLY the provided HR policy context.

Rules:

1. Use the provided context as your primary and only source.
2. Do not invent HR policies, rules, benefits, leave limits, salaries,
   penalties, eligibility requirements, or procedures.
3. If the answer cannot be found in the provided context, clearly say:
   "I could not find this information in the uploaded HR policy."
4. When possible, mention the relevant policy page number.
5. Keep answers clear and practical.
6. If the policy contains an exception or condition, mention it.
7. Do not present general HR knowledge as if it came from the uploaded PDF.
"""

    user_prompt = f"""
HR POLICY CONTEXT:

{context}

QUESTION:

{question}

Answer the question based strictly on the HR policy context.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.1,
        max_tokens=1000
    )

    return response.choices[0].message.content


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("📄 HR Policy")

    uploaded_file = st.file_uploader(
        "Upload HR Policy PDF",
        type=["pdf"]
    )

    st.divider()

    st.markdown("### How it works")

    st.markdown(
        """
        1. Upload your HR Policy PDF.
        2. The PDF is extracted using PyMuPDF.
        3. Text is divided into chunks.
        4. Sentence Transformers creates embeddings.
        5. FAISS stores the embeddings.
        6. Your question is converted into an embedding.
        7. Relevant policy sections are retrieved.
        8. Groq generates the final answer.
        """
    )

    st.divider()

    st.caption(
        "RAG = Retrieval-Augmented Generation"
    )


# --------------------------------------------------
# INITIALIZE SESSION STATE
# --------------------------------------------------

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_name" not in st.session_state:
    st.session_state.file_name = None


# --------------------------------------------------
# PROCESS PDF
# --------------------------------------------------

if uploaded_file:

    if st.session_state.file_name != uploaded_file.name:

        with st.spinner("Processing HR Policy PDF..."):

            try:

                pages = extract_pdf_text(uploaded_file)

                if not pages:
                    st.error(
                        "No readable text was found in this PDF. "
                        "If this is a scanned PDF, OCR may be required."
                    )
                    st.stop()

                chunks = create_chunks(pages)

                if not chunks:
                    st.error(
                        "Could not create text chunks from the PDF."
                    )
                    st.stop()

                index = create_faiss_index(chunks)

                st.session_state.index = index
                st.session_state.chunks = chunks
                st.session_state.file_name = uploaded_file.name
                st.session_state.messages = []

                st.success(
                    f"PDF processed successfully: "
                    f"{len(pages)} pages, {len(chunks)} chunks."
                )

            except Exception as e:

                st.error(
                    f"Error while processing PDF: {str(e)}"
                )
                st.stop()


# --------------------------------------------------
# MAIN AREA
# --------------------------------------------------

if st.session_state.index is None:

    st.info(
        "👈 Upload an HR Policy PDF from the sidebar to get started."
    )

    st.markdown(
        """
        ### Example questions

        - What is the annual leave policy?
        - How many sick leaves are employees entitled to?
        - What is the probation period?
        - What is the maternity leave policy?
        - What are the working hours?
        - What is the resignation notice period?
        - How does the performance appraisal process work?
        """
    )

else:

    st.success(
        f"📄 Active document: **{st.session_state.file_name}**"
    )

    # Display previous conversation
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if (
                message["role"] == "assistant"
                and "sources" in message
            ):

                with st.expander("📚 View sources"):

                    for source in message["sources"]:

                        st.markdown(
                            f"**Page {source['page']}** "
                            f"(similarity: {source['score']:.3f})"
                        )

                        st.caption(
                            source["text"][:500] + "..."
                            if len(source["text"]) > 500
                            else source["text"]
                        )

    # Chat input
    question = st.chat_input(
        "Ask something about the HR policy..."
    )

    if question:

        # Add user question
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        # Retrieve relevant policy sections
        with st.spinner("Searching the HR policy..."):

            retrieved_chunks = retrieve_chunks(
                question,
                st.session_state.index,
                st.session_state.chunks,
                top_k=5
            )

        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner("Generating answer..."):

                try:

                    answer = generate_answer(
                        question,
                        retrieved_chunks
                    )

                    st.markdown(answer)

                    with st.expander("📚 View sources"):

                        for source in retrieved_chunks:

                            st.markdown(
                                f"**Page {source['page']}** "
                                f"(similarity: "
                                f"{source['score']:.3f})"
                            )

                            st.caption(
                                source["text"][:500] + "..."
                                if len(source["text"]) > 500
                                else source["text"]
                            )

                    # Save assistant response
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": retrieved_chunks
                        }
                    )

                except Exception as e:

                    st.error(
                        f"Error generating answer: {str(e)}"
                    )
