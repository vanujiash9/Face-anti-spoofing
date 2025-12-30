import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="FAS Multi-GPU Runner")
    parser.add_argument("--model", type=str, required=True, choices=['convnext', 'efficientnet', 'vit'])
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    if args.model == 'convnext':
        from src.training.train_convnext import main as run
    elif args.model == 'efficientnet':
        from src.training.train_efficientnet import main as run
    elif args.model == 'vit':
        from src.training.train_vit import main as run
    
    run()

if __name__ == "__main__":
    main()