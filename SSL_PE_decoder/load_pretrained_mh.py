import torch
import torchvision
import tqdm
import argparse
import objgraph
import time
import numpy
import math

from dn3_ext_mh import BENDRClassification, LinearHeadBENDR
import dn3_utils_mh as utils
from processes_mh import StandardClassification


print(torch.cuda.is_available())
model_enc = torch.load('encoder.pt')
model_con = torch.load('contextualizer.pt')
print("END")