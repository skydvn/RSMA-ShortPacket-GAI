import argparse
from pathlib import Path


def get_arguments():
    parser = argparse.ArgumentParser()
    """ ======================================================== """
    """ ====================== Run config ===================== """
    """ ======================================================== """
    parser.add_argument("--seed", type=int, default=730,
                        help="one manual random seed")
    parser.add_argument("--n-seed", type=int, default=1,
                        help="number of runs")

    # --------------------- Path
    parser.add_argument("--data-dir", type=Path, default="D:/Datasets/",
                        help="Path to the mnist dataset")
    parser.add_argument("--exp-dir", type=Path, default="D:/Github/1-RepresentationLearning/IVAE/experiments",
                        help="Path to the experiment folder, where all logs/checkpoints will be stored")
    parser.add_argument("--flag-d-collect", type=bool, default=True,
                        help="Trigger for training")
    parser.add_argument("--flag-train", type=bool, default=False,
                        help="Trigger for training")
    parser.add_argument("--flag-eval", type=bool, default=False,
                        help="Trigger for evaluation")

    """ ======================================================== """
    """ ====================== Flag & name ===================== """
    """ ======================================================== """
    parser.add_argument("--mode", type=str, default="train",
                        help="experiment mode")
    parser.add_argument("--log-delay", type=float, default=2.0,
                        help="Time between two consecutive logs (in seconds)")
    parser.add_argument("--eval", type=bool, default=True,
                        help="Evaluation Trigger")
    parser.add_argument("--log-flag", type=bool, default=False,
                        help="Logging Trigger")
    parser.add_argument("--f-cluster", type=bool, default=True,
                        help="Trigger the clustering to get salient feature of specific categories")
    parser.add_argument("--plot-interval", type=int, default=5000,
                        help="Number of step needed to plot new accuracy plot")
    parser.add_argument("--save-flag", type=bool, default=True,
                        help="Save Trigger")

    """ ======================================================== """
    """ ================== Environment config ================== """
    """ ======================================================== """
    parser.add_argument("--antenna-num", type=int, default=8,
                        help="number of antenna")
    parser.add_argument("--user-num", type=int, default=2,
                        help="number of users")
    parser.add_argument("--bandwidth", type=float, default=100*1e6,
                        help="signal bandwidth")
    parser.add_argument("--nfdb", type=float, default=6.5,
                        help="NFdB")
    parser.add_argument("--carrier-freq", type=float, default=300*1e9,
                        help="carrier frequency in GHz (300 GHz)")
    parser.add_argument("--kabs", type=float, default=0.0033,
                        help="absorption loss coefficient measured at 300 GHz")

    parser.add_argument("--tx-db", type=float, default=3,
                        help="rx antenna gain in dB")
    parser.add_argument("--rx-db", type=float, default=0,
                        help="rx antenna gain in dB")

    parser.add_argument("--radius-in", type=float, default=0,
                        help="radius of cell-in")
    parser.add_argument("--radius-out", type=float, default=10,
                        help="radius of cell-out")

    parser.add_argument("--alloc-common", type=float, default=0.7,
                        help="power allocation for common packet")

    parser.add_argument("--rate-common", type=float, default=0.5,
                        help="rate for common packet")
    parser.add_argument("--rate-private", type=float, default=0.75,
                        help="rate for private packet")

    parser.add_argument("--imperfect-sic", type=float, default=0.1,
                        help="imperfect SIC coefficient")


    """ ======================================================== """
    """ ===================== Agent config ===================== """
    """ ======================================================== """
    parser.add_argument("--algo", type=str, default="LICE",
                        help="name of algorithm")

    parser.add_argument("--memory-size", type=int, default=100000,
                        help="size of the replay memory")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="data batch size")

    parser.add_argument("--initial-steps", type=int, default=1e4,
                        help="initial random steps")

    parser.add_argument("--max-episode", type=int, default=100,
                        help="max episode")
    parser.add_argument("--max-step", type=int, default=500,
                        help="max number of step per episode")
    parser.add_argument("--trial", type=int, default=1,
                        help="number of trial (samplings)")

    # Learning Rates
    parser.add_argument('--learning-rate', '-lr', default=1e-4, type=float, help='Learning rate')

    # Hyperparams
    parser.add_argument('--beta', default=1., type=float, help='Beta vae beta')
    parser.add_argument('--capacity', default=None, type=float, help='KL Capacity')
    parser.add_argument('--capacity-leadin', default=100000, type=int, help='KL capacity leadin')

    return parser.parse_args()
