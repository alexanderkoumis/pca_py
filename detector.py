# coding=utf-8
import abc
import os
import xml.etree.ElementTree

import cv2
import numpy


class _PCADetector(object):

    train_M = 0
    train_N = 256
    train_NN = train_N * train_N
    train_images = []
    train_col_vecs = []
    train_projections = []

    __metaclass__ = abc.ABCMeta

    def __init__(self, train_dir, test_dir):
        self.train_dir = train_dir
        self.test_dir = test_dir

    @abc.abstractmethod
    def _set_images(self):
        raise NotImplementedError

    def _run(self):
        # Steps from http://www.vision.jhu.edu/teaching/vision08/Handouts/case_study_pca1.pdf
        # Good guide: http://www.mathworks.com/matlabcentral/fileexchange/45750-face-recognition-using-pca
        # Good paper: http://www.ece.lsu.edu/gunturk/EE7700/Eigenface.pdf

        # Step 1: Obtain training face images I1, I2, ..., IM
        self._set_images()

        # Step 2: Represent every image I as a flattened vector, Γ
        train_gamma = numpy.zeros((self.train_NN, self.train_M))
        for col in range(0, self.train_M - 1):
            train_gamma[:, col] = self.train_images[col].ravel()

        # Step 3: Compute the mean face vector, Ψ
        train_psi = numpy.mean(train_gamma, axis=1)

        # Step 4: Subtract the mean face, getting Φ [A = Φ1, Φ2, ..., ΦM]
        train_A = train_gamma - train_psi.reshape(self.train_NN, 1)

        # Step 5: Compute the covariance matrix, C

        # Step 6: Compute the Eigenvectors ui of C, A(A^T)

        # Because 5 and 6 result in N2xN2 matrix, we will instead compute eigenvectors of L ((A^T)A), vi, whose
        # eigenvectors are linearly related to ui by: ui = Avi

        # Step 6.1: Compute L, (A^T)A
        train_L = train_A.T.dot(train_A)

        # Step 6.2: Compute eigenvectors vi of (A^T)A
        train_w, train_v = numpy.linalg.eig(train_L)

        #   Step 6.3: Compute M best eigenvectors/eigenfaces of A(A^T) : ui = Avi, normalize such that ||ui|| = 1
        train_u = train_A.dot(train_v)
        train_u /= numpy.linalg.norm(train_u)

        self.train_projections = numpy.zeros((self.train_M, self.train_NN))
        for i in range(0, self.train_M - 1):
            projected_face = train_u.dot(train_A[i].T)
            self.train_projections[i] = projected_face

    def show_images(self, display_type='projected'):
        display_images = self.train_projections if display_type == 'projected' else self.train_images
        for display_image in display_images:
            cv2.imshow('lol', display_image.reshape(self.train_N, self.train_N))
            cv2.waitKey(33)


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
                    self.train_images.append(cv2.resize(image_cropped, (self.train_N, self.train_N)))

        self.train_M = len(self.train_images)

