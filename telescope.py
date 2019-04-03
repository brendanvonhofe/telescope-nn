'''
Command line tool to crop subjects out of images with a trimap.
Usage: python telescope.py PATH/TO/IMAGE PATH/TO/TRIMAP
Output is a one channel image (alpha-matte) of the subject.
'''
import numpy as np
from skimage import io
import sys
import os

sys.path.insert(0, 'src')

from predict import predict

def main():
    # Get crop prediction
    output = predict(sys.argv[1], sys.argv[2], os.getcwd())

    # Save output for viewing
    io.imsave("alpha_matte.png", output)
    print("Output saved to \"./alpha_matte.png\"")

if __name__ == '__main__':
    main()
