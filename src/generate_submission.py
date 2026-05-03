import argparse

from src.bis_retriever import generate_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate BIS standard retrieval output in hackathon submission format"
    )
    parser.add_argument(
        "--test_set",
        type=str,
        required=True,
        help="Path to input queries JSON",
    )
    parser.add_argument(
        "--dataset_pdf",
        type=str,
        default="dataset.pdf",
        help="Path to BIS dataset PDF",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission_output.json",
        help="Path to output JSON",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of standards to retrieve per query",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="index_cache.json",
        help="Path for cached extracted PDF index",
    )
    parser.add_argument(
        "--context_multiplier",
        type=float,
        default=1.7,
        help="Context boost multiplier",
    )
    parser.add_argument(
        "--method_penalty",
        type=float,
        default=0.45,
        help="Method penalty for 'methods of test' standards",
    )
    parser.add_argument(
        "--explicit_boost",
        type=float,
        default=12.0,
        help="Boost for explicitly mentioned standards",
    )

    args = parser.parse_args()

    generate_results(
        test_set_path=args.test_set,
        dataset_pdf_path=args.dataset_pdf,
        output_path=args.output,
        top_k=args.top_k,
        cache_path=args.cache,
        context_multiplier=args.context_multiplier,
        method_penalty=args.method_penalty,
        explicit_boost=args.explicit_boost,
    )


if __name__ == "__main__":
    main()
