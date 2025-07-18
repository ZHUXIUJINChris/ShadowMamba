import numpy as np
import os
import argparse

from skimage.util import img_as_ubyte
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils

from utils.loader import get_test_data
from Localmamba import LocalMamba
from Shadowmamba import ShadowMamba

from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Image Deshadow using MPRDNet')


parser.add_argument('--input_dir', default='', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default=r'', type=str, help='Directory for results')
parser.add_argument('--weights', default= r'', type=str, help='Path to weights')
parser.add_argument('--dataset', default='AISTD', type=str, help='Test Dataset')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--datasetname', default='epoch_best', type=str, help='Test Dataset')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

model_deshadow = ShadowMamba()

utils.load_checkpoint(model_deshadow,args.weights)
print("===>Testing using weights: ",args.weights)
model_deshadow.cuda()
model_deshadow = nn.DataParallel(model_deshadow)
model_deshadow.eval()

dataset = args.dataset

test_dataset = get_test_data(args.input_dir)

#datasetname = args.datasetname
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)
result_dir  = os.path.join(args.result_dir, dataset)
utils.mkdir(result_dir)

with torch.no_grad():
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        input     = data_test[0].cuda()
        mask       = data_test[1].cuda()

        filenames = data_test[2]

        restored = model_deshadow(input, mask)
        restored = torch.clamp(restored,0,1)

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

        for batch in range(len(restored)):
            restored_img = img_as_ubyte(restored[batch])
            utils.save_img((os.path.join(result_dir, filenames[batch])), restored_img)