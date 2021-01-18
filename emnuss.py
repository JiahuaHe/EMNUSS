'''
Copyright (C) 2020 Jiahua He, Sheng-You Huang and Huazhong University of Science and Technology

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
import os
import sys
import json
import torch
import argparse
import numpy as np
from torch import nn
from tqdm import tqdm
from math import ceil
from torch import FloatTensor as FT
from torch.nn.functional import softmax
from torch.autograd import Variable as V

from models import NestedUNet
from utils import *

def get_args():
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", "-c", type=str, default="./config.json", help="json config file")
    parser.add_argument("--mapIn", "-mi", type=str, required=True, help="Input EM density map file")
    parser.add_argument("--mapOut", "-mo", type=str, required=True, help="Output predicted secondary structure map file")
    parser.add_argument("--threshold", "-t", type=float, required=True, help="Contour level")
    parser.add_argument("--output", action="store_true", default=False, help="Write secondary structure prediction into individual maps (helix.mrc, sheet.mrc, coil.mrc)")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    mapIn = args.mapIn
    mapOut = args.mapOut
    threshold = args.threshold
    output = args.output

    with open(args.config) as f:
        config = json.load(f)

    test_params = config['test_params']
    use_gpu = test_params['use_gpu']
    gpu_id = test_params['gpu_id']
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    if use_gpu:
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            print("# Running on {} GPU(s).".format(n_gpus))
        else:
            print("CUDA not available.")
            sys.exit()
    else:
        n_gpus = 0
        print("# Running on CPU.")

    model_file = test_params['model_file']
    batch_size = test_params['batch_size']

    ''' data_params '''
    data_params = config['data_params']
    apix = data_params['apix']
    ignorestart = data_params['ignorestart']
    chunk_size = data_params['chunk_size']
    box_size = data_params['box_size']
    core_size = data_params['core_size']
    voxel_size = [apix, apix, apix]

    ''' net_params ''' 
    net_params = config['net_params']
    in_channels = net_params['in_channels']
    out_channels_first_conv = net_params['out_channels_first_conv']
    n_classes = net_params['n_classes']

    print("# Load map data")    
    map, origin, cella, nxyz, _ = parse_map(mapIn, ignorestart, apix=apix)

    chunks, ncx, ncy, ncz = split_map_into_chunks(map, chunk_size, core_size, dtype="float32", padding=0.0)
    cella = np.asarray([ncz * core_size * apix, ncy * core_size * apix, ncx * core_size * apix], dtype='float32') # rescale cella
    n_chunks = len(chunks)
    print("# Split map into {} chunk(s)".format(n_chunks))

    ''' 3 for region with density below contour level '''
    below_threshold = np.where(map <= threshold, 3.0, 0.0)
    below_threshold = pad_map(below_threshold, box_size, core_size, dtype="float32", padding=3.0)

    ''' normalize chunks and del non-positive chunks '''
    del_indices = []
    
    chunks_norm = np.zeros((n_chunks, box_size, box_size, box_size), dtype=np.float32)
    for i, chunk in enumerate(chunks):
        maximum = chunk.max()
        if maximum <= threshold:
            del_indices.append(i)
            continue
        chunk_norm = chunk.clip(min=0.0) / maximum
        chunks_norm[i], = center_crop(chunk_size, box_size, chunk_norm)

    chunks = np.delete(chunks_norm, del_indices, axis=0)
    keep_indices = np.delete(np.arange(n_chunks, dtype=np.int), del_indices, axis=0)
    n_chunks0 = len(chunks)

    X = V(FT(chunks), requires_grad=False).view(-1, 1, box_size, box_size, box_size) 
    del chunks

    data_num = len(X)

    n_batches = ceil(data_num/batch_size)

    model = NestedUNet(in_channels = in_channels, 
                       out_channels_first_conv = out_channels_first_conv,
                       n_classes = n_classes
                       )
    model_state_dict = torch.load(model_file)
    model.load_state_dict(model_state_dict)
    if use_gpu:
        torch.cuda.empty_cache()
        model = model.cuda()
        if n_gpus > 1:
            model = nn.DataParallel(model)
       
    model.eval()

    chunks_pred0 = np.zeros((n_chunks0, box_size, box_size, box_size), dtype="float32")

    print("# Start prediction")
    for i in tqdm(range(n_batches)):
        X_batch = X[i * batch_size : (i + 1) * batch_size]
        if use_gpu:
            X_batch = X_batch.cuda()
        
        y_pred = model(X_batch)
        y_pred = softmax(y_pred, dim=1)
        y_pred = y_pred.cpu().detach().numpy()

        y_pred_argmax = np.argmax(y_pred, axis=1) 

        chunks_pred0[i * batch_size : (i + 1) * batch_size] = y_pred_argmax

    chunks_pred = np.full((n_chunks, box_size, box_size, box_size), 3.0, dtype="float32")
    chunks_pred[keep_indices] = chunks_pred0
    del chunks_pred0
    map_pred = get_map_from_chunks(chunks_pred, ncx, ncy, ncz, box_size, core_size, dtype=np.float32)
    map_pred = np.where(below_threshold > map_pred, below_threshold, map_pred)
    write_map(mapOut, map_pred, origin, cella, voxel_size)

    if output:
        map_helix = np.where(map_pred == 0.0, 1.0, 0.0).astype(np.float32)
        map_sheet = np.where(map_pred == 1.0, 1.0, 0.0).astype(np.float32)
        map_other = np.where(map_pred == 2.0, 1.0, 0.0).astype(np.float32)

        write_map("helix.mrc", map_helix, origin, cella, voxel_size)
        write_map("sheet.mrc", map_sheet, origin, cella, voxel_size)
        write_map("coil.mrc", map_other, origin, cella, voxel_size)

if __name__ == "__main__":
    main()
