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
import sys
import mrcfile
import numpy as np
from math import ceil

from interp3d import interp3d

def center_crop(chunk_size, box_size, *inputs):
    start = int((chunk_size - box_size) / 2)
    outputs = []
    for input in inputs:
        assert(np.shape(input) == (chunk_size, chunk_size, chunk_size))
        output = input[start : start + box_size,
                       start : start + box_size,
                       start : start + box_size]
        outputs.append(output)
    return outputs

def split_map_into_chunks(map, box_size, core_size, dtype="float32", padding=0):
    map_shape = np.shape(map)

    padded_map = np.full((map_shape[0] + 2 * box_size, map_shape[1] + 2 * box_size, map_shape[2] + 2 * box_size), padding, dtype=dtype)
    padded_map[box_size : box_size + map_shape[0], box_size : box_size + map_shape[1], box_size : box_size + map_shape[2]] = map

    chunk_list = list()

    start_point = box_size - int((box_size - core_size) / 2)
    cur_x, cur_y, cur_z = start_point, start_point, start_point
    while (cur_z + (box_size - core_size) / 2 < map_shape[2] + box_size):
        next_chunk = padded_map[cur_x:cur_x + box_size, cur_y:cur_y + box_size, cur_z:cur_z + box_size]

        cur_x += core_size
        if (cur_x + (box_size - core_size) / 2 >= map_shape[0] + box_size):
            cur_y += core_size
            cur_x = start_point # Reset
            if (cur_y + (box_size - core_size) / 2  >= map_shape[1] + box_size):
                cur_z += core_size
                cur_y = start_point # Reset
                cur_x = start_point # Reset

        chunk_list.append(next_chunk)

    n_chunks = len(chunk_list)
    ncx, ncy, ncz = [ceil(map_shape[i] / core_size) for i in range(3)]
    assert(n_chunks == ncx * ncy * ncz)

    chunks = np.asarray(chunk_list, dtype=dtype)
    return chunks, ncx, ncy, ncz

def pad_map(map, box_size, core_size, dtype="float32", padding=0):
    map_shape = np.shape(map)
    ncx, ncy, ncz = [ceil(map_shape[i] / core_size) for i in range(3)]

    padded_map = np.full((ncx * core_size, ncy * core_size, ncz * core_size), padding, dtype=dtype)
    padded_map[:map_shape[0], :map_shape[1], :map_shape[2]] = map
    
    return padded_map
    
def get_map_from_chunks(chunks, ncx, ncy, ncz, box_size, core_size, dtype='float32'):
    map = np.zeros((ncx * core_size, ncy * core_size, ncz * core_size), dtype=dtype)
    start = int((box_size - core_size) / 2)
    end = int((box_size - core_size) / 2) + core_size
    i = 0
    for z_steps in range(ncz):
        for y_steps in range(ncy):
            for x_steps in range(ncx):
                map[x_steps * core_size : (x_steps + 1) * core_size,
                    y_steps * core_size : (y_steps + 1) * core_size,
                    z_steps * core_size : (z_steps + 1) * core_size] = chunks[i, start : end, start : end, start : end]
                i += 1

    return map 

def parse_map(map_file, ignorestart, apix=None):

    ''' parse mrc '''
    mrc = mrcfile.open(map_file, mode='r')

    map = np.asfarray(mrc.data.copy(), dtype=np.float32)
    voxel_size = np.asarray([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z], dtype=np.float32)
    ncrsstart = np.asarray([mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart], dtype=np.float32)
    origin = np.asarray([mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z], dtype=np.float32)
    cella = (mrc.header.cella.x, mrc.header.cella.y, mrc.header.cella.z)
    ncrs = (mrc.header.nx, mrc.header.ny, mrc.header.nz)
    angle = np.asarray([mrc.header.cellb.alpha, mrc.header.cellb.beta, mrc.header.cellb.gamma], dtype=np.float32)

    ''' check orthogonal '''
    try:
        assert(angle[0] == angle[1] == angle[2] == 90.0)
    except AssertionError:
        print("# Input grid is not orthogonal. EXIT.")
        mrc.close()
        sys.exit()

    ''' reorder axes '''
    mapcrs = np.subtract([mrc.header.mapc, mrc.header.mapr, mrc.header.maps], 1)
    sort = np.asarray([0, 1, 2], dtype=np.int)
    for i in range(3):
        sort[mapcrs[i]] = i
    nxyzstart = np.asarray([ncrsstart[i] for i in sort])
    nxyz = np.asarray([ncrs[i] for i in sort])

    map = np.transpose(map, axes=2-sort[::-1])
    mrc.close()

    ''' shift origin according to n*start '''
    if not ignorestart:
        origin += np.multiply(nxyzstart, voxel_size)

    ''' interpolate grid interval '''
    if apix is not None:
        try:
            assert(voxel_size[0] == voxel_size[1] == voxel_size[2] == apix)
        except AssertionError:
            target_voxel_size = np.asarray([apix, apix, apix], dtype=np.float32)
            print("# Rescale voxel size from {} to {}".format(voxel_size, target_voxel_size))
            interp3d.linear(map, voxel_size[2], voxel_size[1], voxel_size[0], apix, nxyz[2], nxyz[1], nxyz[0])
            map = interp3d.mapout
            nxyz = np.asarray([interp3d.pextx, interp3d.pexty, interp3d.pextz], dtype=np.int)
            voxel_size = target_voxel_size

    assert(np.all(nxyz == np.asarray([map.shape[2], map.shape[1], map.shape[0]], dtype=np.int)))

    return map, origin, cella, nxyz, voxel_size

def write_map(file_name, map, origin, cella, voxel_size):
    mrc = mrcfile.new(file_name, overwrite=True)
    mrc.set_data(map)
    (mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart) = (0, 0, 0)
    (mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z) = origin
    (mrc.header.cella.x, mrc.header.cella.y, mrc.header.cella.z) = cella
    mrc.voxel_size = [voxel_size[i] for i in range(3)]

    mrc.close()
