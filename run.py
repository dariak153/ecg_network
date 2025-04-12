import argparse
import yaml
from training import evaluate
from training import train 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--mode', default='train', choices=['train', 'evaluate'])
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.mode == 'train':
        train.main(config)
    elif args.mode == 'evaluate':
        evaluate.main(config)

if __name__ == "__main__":
    main()
