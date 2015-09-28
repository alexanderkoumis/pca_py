import argparse
# import cv2
from detector import ColorFeretFaceDetector


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('train_path', help='Specify location of training images')
    parser.add_argument('test_path', help='Specify location of test images')
    parser.add_argument('--dataset', help='Specify dataset type (Currently supported: colorferet)')
    args = parser.parse_args()

    if args.dataset == 'colorferet':
        detector = ColorFeretFaceDetector(args.train_path, args.test_path)
        detector.navigate_images()

    # video_capture = cv2.VideoCapture(0)
    #
    # while True:
    #     ret, frame = video_capture.read()
    #
    #     if ret:
    #         cv2.imshow('webcam', frame)
    #         key = cv2.waitKey(33)
    #         if key >= 0:
    #             break
    #     else:
    #         print 'Error reading video'
    #         break


if __name__ == "__main__":
    main()

