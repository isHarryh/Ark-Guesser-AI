import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Ark Guesser AI Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subcommand: dataset_generate
    dataset_parser = subparsers.add_parser("dataset_generate", help="Generate dataset from images")
    dataset_parser.add_argument("image_dir", help="Directory containing images")
    dataset_parser.add_argument("output_path", help="Output path for the dataset JSON file")
    dataset_parser.add_argument("-p", "--num_processes", type=int, default=1, help="Number of processes for generation")
    dataset_parser.add_argument(
        "--include-ranking",
        action="store_true",
        help="Contain additional human ranking information, "
        "when enabled, the output dataset will contain human performance data for evaluation purposes",
    )

    # Subcommand: train
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("dataset_path", help="Path to the dataset JSON file")
    train_parser.add_argument("model_path", help="Path to save the trained model")

    # Subcommand: eval
    eval_parser = subparsers.add_parser("eval", help="Evaluate the model")
    eval_parser.add_argument("dataset_path", help="Path to the evaluation dataset JSON file")
    eval_parser.add_argument("model_path", help="Path to the trained model")

    # Subcommand: infer
    infer_parser = subparsers.add_parser("infer", help="Run inference on image files")
    infer_parser.add_argument("dataset_path", help="Path to dataset JSON file")
    infer_parser.add_argument("model_path", help="Path to trained model")
    infer_parser.add_argument("image_path", help="Image file or directory")

    args = parser.parse_args()

    if args.command == "dataset_generate":
        from src.dataset_generator import main as dataset_generate_main

        dataset_generate_main(
            args.image_dir,
            args.output_path,
            include_ranking=args.include_ranking,
            num_processes=args.num_processes,
        )

    elif args.command == "train":
        from src.train import main as train_main

        train_main(args.dataset_path, args.model_path)

    elif args.command == "eval":
        from src.eval import main as eval_main

        eval_main(args.dataset_path, args.model_path)

    elif args.command == "infer":
        from src.infer import main as infer_main

        infer_main(args.dataset_path, args.model_path, args.image_path)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
