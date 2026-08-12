import argparse


def main():
    parser = argparse.ArgumentParser(
        description="AAPL Decoupling Prediction System"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db")
    sub.add_parser("features-audit")
    sub.add_parser("labels-audit")
    sub.add_parser("test")
    sub.add_parser("history-eval")
    sub.add_parser("deploy")

    for name in ["fetch", "summary", "align", "predict", "live-eval"]:
        command = sub.add_parser(name)
        command.add_argument("date", help="YYYY-MM-DD")

    args = parser.parse_args()

    if args.command == "init-db":
        from src.database import initialize
        initialize()

    elif args.command == "fetch":
        from src.data import fetch_day
        fetch_day(args.date)

    elif args.command == "summary":
        from src.data import show_day_summary
        show_day_summary(args.date)

    elif args.command == "align":
        from src.data import build_aligned_master
        df = build_aligned_master(args.date)
        print(
            df[["Datetime", "AAPL_Close", "QQQ_Close", "SPY_Close"]]
            .head()
            .to_string(index=False)
        )

    elif args.command == "features-audit":
        from src.features import audit_historical_features
        audit_historical_features()

    elif args.command == "labels-audit":
        from src.labeling import audit_historical_labels
        audit_historical_labels()

    elif args.command == "test":
        from src.model import run_frozen_test
        run_frozen_test()

    elif args.command == "history-eval":
        from src.model import evaluate_historical_splits
        evaluate_historical_splits()

    elif args.command == "deploy":
        from src.model import train_deployment_model
        train_deployment_model()

    elif args.command == "predict":
        from src.pipeline import run_prediction_pipeline
        run_prediction_pipeline(args.date)

    elif args.command == "live-eval":
        from src.model import evaluate_live_day
        evaluate_live_day(args.date)


if __name__ == "__main__":
    main()