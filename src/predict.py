'''
Use full neural network to predict image matte for subject given a trimap.

Currently has to load all the weights everytime prediction is done.
'''

from pathlib import Path
import torch
import numpy as np
from skimage import io
import sys

sys.path.insert(0, 'src')

from architecture.vgg16 import DeepMattingVGG
from architecture.refinement_layer import MatteRefinementLayer

def predict(image_fn, trimap_fn, home_dir, use_trimap=True):
    # Path to model files
    ENCDEC = home_dir + '/model/ckpt_encdec_e1281.pth'
    REFINE = home_dir + '/model/ckpt_refinement_e1281.pth'

    # Define to do computations on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load main network and refinement network as well as their pretrained weights
    net = DeepMattingVGG()
    net.load_state_dict(torch.load(ENCDEC)['state_dict'])
    net = net.to(device)
    refine = MatteRefinementLayer()
    refine.load_state_dict(torch.load(REFINE)['state_dict'])
    refine = refine.to(device)


    # Read in images
    im, tri = io.imread(image_fn).astype(np.float)*(1./255), io.imread(trimap_fn).astype(np.float)*(1./255)
    if(len(tri.shape) < 3):
        tri = np.expand_dims(tri, -1)

    # Convert images to torch objects and feed to network
    image, trimap = torch.FloatTensor(im.transpose((2,0,1))).unsqueeze(0).to(device), torch.FloatTensor(tri.transpose((2,0,1))).unsqueeze(0).to(device)
    inputs = torch.cat((image, trimap), 1)
    output = (refine(torch.cat((image, net(inputs)), 1)).cpu().detach().numpy()*255).astype(np.uint8)[0][0]

    # Use information from trimap to clean up prediction
    if(use_trimap):
        tri = tri[:,:,0]
        output[np.equal(tri, 1).astype(np.bool)] = 255
        output[np.equal(tri, 0).astype(np.bool)] = 0

    return output