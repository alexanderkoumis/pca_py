import argparse
from detector import ColorFeretFaceDetector


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('train_path', help='Specify location of training images')
    parser.add_argument('test_path', help='Specify location of test images')
    parser.add_argument('--dataset', help='Specify dataset type (Currently supported: colorferet)')
    args = parser.parse_args()

    if args.dataset == 'colorferet':
        detector = ColorFeretFaceDetector(args.train_path, args.test_path)
        detector.show_images('projected')


if __name__ == "__main__":
    main()

