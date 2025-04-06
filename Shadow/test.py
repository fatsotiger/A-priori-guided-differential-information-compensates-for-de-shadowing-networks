import os

import torch
import argparse
from torch.utils.data import DataLoader

from model_convnext import fusion_net
from torchvision.utils import save_image as imwrite, save_image
from DataSet.DataSetTest import DataSet_t
parser = argparse.ArgumentParser(description='Dehaze')
parser.add_argument('--test_dir', type=str, default='./Data/ntire25_sh_rem_test_inp//')
args = parser.parse_args()
path = args.test_dir  # 'D:\Data\data_track2_phase2_paired_LR\\'
test_dataset = DataSet_t(path)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)


# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
net = fusion_net()

# --- Multi-GPU --- #
net = net.cuda()
# net = nn.DataParallel(net)
try:
    net.load_state_dict(torch.load('check_points/epochbest.pkl'))
    print('..')
except:
    print('.')
# --- Test --- #
with torch.no_grad():
    net.eval()
    for batch_idx, (hazy,name) in enumerate(test_loader):
        hazy = hazy.cuda()
        output = net(hazy)

        save_image(output, 'outputs/' + name[0])  # dehaze_image png













