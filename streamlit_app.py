import streamlit as st
from src.wiki_loader import WikiSectionLoader
from src.retriever import SectionRetriever
from src.answering import generate_grounded_answer


st.set_page_config(page_title="Manchester United Wikipedia RAG", layout="wide")
st.title("Manchester United Wikipedia RAG")
st.write("Ask a question. The app retrieves the most relevant Wikipedia sections and answers only from those sections.")


@st.cache_resource(show_spinner=True)
def load_pipeline():
    loader = WikiSectionLoader()
    documents = loader.load_documents()
    retriever = SectionRetriever()
    retriever.build_index(documents)
    return documents, retriever


documents, retriever = load_pipeline()
st.caption(f"Loaded {len(documents)} section documents from the Manchester United Wikipedia page.")

query = st.text_input("Question", value="When did Manchester United move to Old Trafford?")
top_k = st.slider("Number of sections to retrieve", min_value=1, max_value=8, value=4)

if query:
    results = retriever.retrieve(query, top_k=top_k)
    answer = generate_grounded_answer(query, results)

    st.subheader("Answer")
    st.markdown(answer)

    st.subheader("Retrieved sections")
    for result in results:
        doc = result["document"]
        with st.expander(f"{result['rank']}. {doc['section']} - score {result['score']:.3f}"):
            st.write(doc["text"][:1500] + ("..." if len(doc["text"]) > 1500 else ""))
            st.link_button("Open Wikipedia section", doc["url"])
