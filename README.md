# PCA Face Detection

This detection module currently supports the [Color FERET](http://www.nist.gov/itl/iad/ig/colorferet.cfm) face dataset.

Video of images being projected to "face space":

[![PCA](http://img.youtube.com/vi/H-r0QUaTIu0/0.jpg)](https://www.youtube.com/watch?v=H-r0QUaTIu0)

## Usage

```bash
python pca.py --dataset colorferet <training images path> <test images path>
```

## Todo

1. ~~Load training datasets~~
  1. ~~[Color FERET](http://www.nist.gov/itl/iad/ig/colorferet.cfm)~~
  2. [BioID](https://www.bioid.com/About/BioID-Face-Database)
2. ~~Preprocess train images (center about eyes, resize)~~
3. ~~Compute training eigenfaces~~
4. Detection (still image)
5. Detection (webcam)
6. Create fork using [aws_emr](https://github.com/mwcarlis/aws_emr)
7. (Nice to have) Recognition (still/webcam)
