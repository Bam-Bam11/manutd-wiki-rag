from src.wiki_loader import WikiSectionLoader
from src.retriever import SectionRetriever
from src.answering import generate_grounded_answer, format_sources


def main() -> None:
    loader = WikiSectionLoader()
    print("Loading Wikipedia sections...")
    documents = loader.load_documents()
    print(f"Loaded {len(documents)} section documents.")

    retriever = SectionRetriever()
    print("Building embedding index...")
    retriever.build_index(documents)

    print("\nAsk questions about Manchester United. Type 'quit' to stop.\n")
    while True:
        query = input("Question: ").strip()
        if query.lower() in {"quit", "exit"}:
            break
        if not query:
            continue

        results = retriever.retrieve(query, top_k=4)
        answer = generate_grounded_answer(query, results)

        print("\n" + answer)
        print("\n" + format_sources(results))
        print("-" * 80)


if __name__ == "__main__":
    main()
