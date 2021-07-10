import os
import os.path as p
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Single-Photon Camera Guided HDR Imaging")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input image directory.")
    parser.add_argument("--out", "-o", type=str, required=True, help="Path to output directory.")
    parser.add_argument("--weights", "-w", type=str, required=True, help="Path to pre-trained neural network weights.")
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    print("input = {}".format(args.input))


if __name__ == "__main__":
    main()
