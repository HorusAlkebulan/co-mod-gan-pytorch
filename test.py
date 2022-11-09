"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""


from multiprocessing.spawn import freeze_support

import cv2
import numpy as np
from tqdm import tqdm

import data
#from models.pix2pix_model import Pix2PixModel
import models
import torch
from options.test_options import TestOptions


def test_using_images():
    opt = TestOptions().parse()

    print(f'INFO: Setting to run using CPU only')
    opt.gpu_ids = []

    dataloader = data.create_dataloader(opt)

    model = models.create_model(opt)
    model.eval()

    for i, data_i in tqdm(enumerate(dataloader)):
        if i * opt.batchSize >= opt.how_many:
            break
        with torch.no_grad():
            generated,_ = model(data_i, mode='inference')
        generated = torch.clamp(generated, -1, 1)
        generated = (generated+1)/2*255
        generated = generated.cpu().numpy().astype(np.uint8)
        img_path = data_i['path']
        for b in range(generated.shape[0]):
            pred_im = generated[b].transpose((1,2,0))
            print('process image... %s' % img_path[b])
            cv2.imwrite(img_path[b], pred_im[:,:,::-1])

if __name__ == "__main__":
    freeze_support()

    test_using_images()