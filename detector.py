import abc
import os
import xml.etree.ElementTree

import cv2
import numpy


class _PCADetector:

    image_dim = 256
    train_images = []
    train_col_vecs = []

    __metaclass__ = abc.ABCMeta

    def __init__(self, train_dir, test_dir):
        self.train_dir = train_dir
        self.test_dir = test_dir

    @abc.abstractmethod
    def _set_images(self):
        raise NotImplementedError

    def _run(self):
        # Steps from http://www.vision.jhu.edu/teaching/vision08/Handouts/case_study_pca1.pdf

        # Step 1: Obtain training face images I1, I2, ..., IM
        self._set_images()

        # Step 2: Represent every image Ii as a flattened vector
        train_images_orig = numpy.zeros((self.image_dim * self.image_dim, len(self.train_images)))
        for col in range(0, len(self.train_images)-1):
            train_images_orig[:,col] = self.train_images[col].ravel()

        # Step 3: Compute the average face vector
        train_images_mean = numpy.mean(train_images_orig, axis=1)

        # Step 4: Subtract the mean face
        train_images_sub_mean = train_images_orig - train_images_mean.reshape(len(train_images_mean), 1)

        # Step 5: Compute the covariance matrix
        train_images_cov = numpy.cov(train_images_sub_mean, train_images_sub_mean.T)

        # Step 6: Compute the Eigenvectors ui of AAT


class ColorFeretFaceDetector(_PCADetector):

    bounds_ratio = 2

    def __init__(self, train_dir, test_dir):
        super(ColorFeretFaceDetector, self).__init__(train_dir, test_dir)
        super(ColorFeretFaceDetector, self)._run()

    def _set_images(self):

        image_dir_root = os.path.join(self.train_dir, 'dvd1', 'data', 'images')
        metadata_path = os.path.join(self.train_dir, 'dvd1', 'data', 'ground_truths', 'xml')

        for image_dir, image_subdirs, files in os.walk(image_dir_root):
            if len(files) == 0:
                print 'Directory', image_dir, ' has no images, skipping'
                continue

            if len(image_subdirs):
                print 'Skipping root directory'
                continue

            image_dir_root = os.path.join(image_dir, files[0])

            if not os.path.isfile(image_dir_root):
                print 'Error:', image_dir_root, 'is not an image, skipping'
                continue

            image_ext = files[0].split('.')[-1]
            image_stem = files[0].split('.')[0]
            xml_path = os.path.join(metadata_path, os.path.basename(image_dir), image_stem + '.xml')

            if not os.path.isfile(xml_path):
                print 'Error:', xml_path, 'is not a xml file, skipping'
                continue

            metadata_root = xml.etree.ElementTree.parse(xml_path).getroot()

            face_traits = {}
            eye_data_exists = True

            # TODO: Refactor this, find better function in xml module to load 'Face'
            for face in metadata_root.iter('Face'):
                if len(face._children) < 5:
                    eye_data_exists = False
                    break
                face_traits = {
                    'left_eye': {
                        'x': int(face._children[4].attrib['x']),
                        'y': int(face._children[4].attrib['y'])
                    },
                    'right_eye': {
                        'x': int(face._children[5].attrib['x']),
                        'y': int(face._children[5].attrib['y'])
                    },
                    'mouth': {
                        'x': int(face._children[7].attrib['x']),
                        'y': int(face._children[7].attrib['y'])
                    }
                }
                break

            if eye_data_exists and image_ext == 'ppm':
                # Load images as grayscale (1: color, 0: grayscale, -1: unchanged)
                image_orig = cv2.imread(image_dir_root, 0)
                if image_orig is not None:
                    w_orig = face_traits['left_eye']['x'] - face_traits['right_eye']['x']
                    h_orig = face_traits['mouth']['y'] - face_traits['left_eye']['y']
                    w_scaled = int(w_orig * self.bounds_ratio)
                    h_scaled = int(h_orig * self.bounds_ratio)
                    tl_x = face_traits['right_eye']['x'] - (w_scaled - w_orig) / 2
                    tl_y = face_traits['right_eye']['y'] - (h_scaled - h_orig) / 2
                    image_cropped = image_orig[tl_y:tl_y+h_scaled, tl_x:tl_x+w_scaled]
                    self.train_images.append(cv2.resize(image_cropped, (self.image_dim, self.image_dim)))
