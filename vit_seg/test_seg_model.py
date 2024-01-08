import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=8, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--amp/--no-amp', default=False, type=bool)
    parser.add_argument('--eval-freq', default=5, type=int)
    parser.add_argument('--batch_size', default=60, type=int,            # 原64
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=40, type=int, help='Number of epochs of training.')
    parser.add_argument('--normalization', default='', type=str, help="""""")
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument("--drop-path", default=0.1, type=float, help="""""")
    parser.add_argument("--dropout", default=0.0, type=float,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--weight-decay', type=float, default=0.9, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--scheduler', default='polynomial', type=str, help="""""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    parser.add_argument('--optimizer', type=str, default='sgd', help="""""")
    parser.add_argument('--decoder', type=str, help="""""")
    parser.add_argument('--backbone', default='vit_small', type=str)
    parser.add_argument('--window-stride', default=8, type=int,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--window-size', default=8, type=int, help='')
    parser.add_argument('--crop-size', default=96, type=int, help='')
    parser.add_argument('--im-size', default=224, type=int, help='')
    parser.add_argument('--num_workers', default=12, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dataset", default="env://", type=str, help="""""")
    parser.add_argument("--log-dir", default='', type=str, help="logging directory")
    parser.add_argument('--resume/--no-resume', default=True, type=bool)
    return parser

def main():
    cfg = get_args_parser()


if __name__ == "__main__":
    main()