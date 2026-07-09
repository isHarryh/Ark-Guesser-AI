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
    train_parser.add_argument(
        "-M", "--model",
        choices=["v1", "v2"],
        default="v1",
        help="Model version to use (default: v1)",
    )

    # Subcommand: gui
    subparsers.add_parser("gui", help="Prepare and launch GUI")

    # Subcommand: eval
    eval_parser = subparsers.add_parser("eval", help="Evaluate the model")
    eval_parser.add_argument("dataset_path", help="Path to the evaluation dataset JSON file")
    eval_parser.add_argument("model_path", help="Path to the trained model")
    eval_parser.add_argument(
        "-M", "--model",
        choices=["v1", "v2"],
        default="v1",
        help="Model version to use (default: v1)",
    )

    # Subcommand: infer
    infer_parser = subparsers.add_parser("infer", help="Run inference on image files")
    infer_parser.add_argument("dataset_path", help="Path to dataset JSON file")
    infer_parser.add_argument("model_path", help="Path to trained model")
    infer_parser.add_argument("image_path", help="Image file or directory")
    infer_parser.add_argument(
        "-M", "--model",
        choices=["v1", "v2"],
        default="v1",
        help="Model version to use (default: v1)",
    )

    # Subcommand: dataset_visualize
    viz_parser = subparsers.add_parser("dataset_visualize", help="Generate dataset HTML report")
    viz_parser.add_argument("dataset_path", help="Path to dataset JSON file")
    viz_parser.add_argument("--host", default="127.0.0.1", help="Host to serve the report")
    viz_parser.add_argument("--port", type=int, default=8050, help="Port to serve the report")
    viz_parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode")

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

        train_main(args.dataset_path, args.model_path, model_version=args.model)

    elif args.command == "gui":
        from src.gui import prepare_deps, start_gui

        if prepare_deps():
            start_gui()

    elif args.command == "eval":
        from src.eval import main as eval_main

        eval_main(args.dataset_path, args.model_path, model_version=args.model)

    elif args.command == "infer":
        from src.infer import main as infer_main

        infer_main(args.dataset_path, args.model_path, args.image_path, model_version=args.model)

    elif args.command == "dataset_visualize":
        from src.dataset_visualizer import main as dataset_visualize_main

        dataset_visualize_main(args.dataset_path, host=args.host, port=args.port, debug=args.debug)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
