'''
Command line tool to crop subjects out of images with a trimap.
Usage: python telescope.py PATH/TO/IMAGE PATH/TO/TRIMAP
Output is a one channel image (alpha-matte) of the subject.

Change ENCDEC and REFINE paths to point to model files.
'''

from pathlib import Path
import torch
import numpy as np
from skimage import io
import sys

sys.path.insert(0, 'src')

from architecture.vgg16 import DeepMattingVGG
from architecture.refinement_layer import MatteRefinementLayer

ENCDEC = 'models/nuds_weighted2/encdec/ckpt_encdec_e1281.pth'
REFINE = 'models/nuds_weighted2/refinement/ckpt_refinement_e1281.pth'

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = DeepMattingVGG()
    net.load_state_dict(torch.load(ENCDEC)['state_dict'])
    net = net.to(device)
    refine = MatteRefinementLayer()
    refine.load_state_dict(torch.load(REFINE)['state_dict'])
    refine = refine.to(device)

    im, tri = io.imread(sys.argv[1]).astype(np.float)*(1./255), io.imread(sys.argv[2]).astype(np.float)*(1./255)
    if(len(tri.shape) < 3):
        tri = np.expand_dims(tri, -1)

    image, trimap = torch.FloatTensor(im.transpose((2,0,1))).unsqueeze(0).to(device), torch.FloatTensor(tri.transpose((2,0,1))).unsqueeze(0).to(device)
    inputs = torch.cat((image, trimap), 1)
    output = (refine(torch.cat((image, net(inputs)), 1)).cpu().detach().numpy()*255).astype(np.uint8)[0][0]

    tri = tri[:,:,0]
    output[np.equal(tri, 1).astype(np.bool)] = 255
    output[np.equal(tri, 0).astype(np.bool)] = 0

    io.imsave("cropped_image.png", output)
    print("Output saved to \"./cropped_image.png\"")

if __name__ == '__main__':
    main()