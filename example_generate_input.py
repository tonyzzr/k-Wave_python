"""
 * @file      example_generate_input.py
 *
 * @author    Filip Vaverka
 *            Faculty of Information Technology
 *            Brno University of Technology
 *            ivaverka@fit.vutbr.cz
 *
 * @brief     Simple example of usage of KWaveInputFile class.
 *
 * @date      21 March 2018, 12:23 (created)
 *            21 March 2018, 12:23 (revised)
 *
 * @copyright Copyright (C) 2017 Filip Vaverka, Jiri Jaros and Bradley Treeby.
 *
 * This file is part of the C++ extension of the [k-Wave Toolbox](http://www.k-wave.org).
 *
 * This file is part of the k-Wave. k-Wave is free software: you can redistribute it and/or modify it under the terms
 * of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
 * more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with k-Wave.
 * If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).
 *
"""

import os
import sys
import argparse

import numpy as np
import h5py
import math

from kwave_input_file import KWaveInputFile
from kwave_data_filters import SpectralDataFilter


def Size(s):
    try:
        x, y, z = map(int, s.split(','))
        return x, y, z
    except:
        raise argparse.ArgumentTypeError('Size must be: Nx, Ny, Nz')


grid_res = (1.0e-4, 1.0e-4, 1.0e-4)

p = argparse.ArgumentParser(description='Generate k-Wave simulation input file.')
p.add_argument('size', metavar='size', type=Size, help="Size of the simulation domain: Nx,Ny,Nz")
p.add_argument('filename', metavar='filename', type=str, help="Name of the input file generated")
p.add_argument('example', metavar='example', type=str, help="Use 'p0_source_input', 'p_source_input', 'p_source_input_many'")

args = p.parse_args()

grid_size = args.size

dt = 0.3 * grid_res[0] / 1500.0
end_time = 2.0 * (grid_size[2] * grid_res[2]) / 1500.0
steps = int(end_time / dt)

kwave_file = KWaveInputFile(grid_size, steps, grid_res, dt, pml_size=(20, 20, 20))

if args.example == 'p0_source_input':
    # Generate homogeneous medium with cube shaped P0 source input
    kwave_file.set_medium_properties(1000, 1500)
    p0 = np.zeros(grid_size)
    p0[int(grid_size[0]/2 - 16) : int(grid_size[0]/2 + 16),
       int(grid_size[1]/2 - 16) : int(grid_size[1]/2 + 16),
       int(grid_size[2]/2 - 16) : int(grid_size[2]/2 + 16)] = 1.0e3
    p0 = SpectralDataFilter.smooth(p0)
    kwave_file.set_p0_source_input(p0)
    
elif args.example == 'p_source_input':
    # Generate homogeneous medium with two plane shaped pressure sources
    # with frequency of 1 MHz and amplitude 10 Pa
    kwave_file.set_medium_properties(1000, 1500)
    source_mask = np.zeros(grid_size)
    source_mask[ int(grid_size[0]/4), int(grid_size[1]/2 - 4):int(grid_size[1]/2 + 4), int(grid_size[2]/2 - 4):int(grid_size[2]/2 + 4)] = 1;
    source_mask[-int(grid_size[0]/4), int(grid_size[1]/2 - 4):int(grid_size[1]/2 + 4), int(grid_size[2]/2 - 4):int(grid_size[2]/2 + 4)] = 1;
    
    # All matrices are internally transposed, therefore we need to transpose 
    # our source mask before computing indexes of the source points.
    p_source_index = np.flatnonzero(source_mask.transpose((2, 1, 0)))
    p_source_index = np.reshape(p_source_index, (p_source_index.size, 1, 1))
    
    FREQ = 1.0e6 # Hz
    AMPL = 10
    p_source_input = np.reshape(AMPL * np.sin(FREQ * (2.0*math.pi) * np.arange(0.0, steps*dt, dt)), (1, steps, 1))
    
    kwave_file.set_p_source(p_source_index, p_source_input)

elif args.example == 'p_source_input_many':
    # Generate homogeneous medium with two plane shaped pressure sources 
    # (with independent signals) with frequency of 1 MHz and amplitude 50 Pa and 100 Pa
    kwave_file.set_medium_properties(1000, 1500)
    source_mask = np.zeros(grid_size)
    source_mask[ int(grid_size[0]/4), int(grid_size[1]/2 - 4):int(grid_size[1]/2 + 4), int(grid_size[2]/2 - 4):int(grid_size[2]/2 + 4)] = 1;
    source_mask[-int(grid_size[0]/4), int(grid_size[1]/2 - 4):int(grid_size[1]/2 + 4), int(grid_size[2]/2 - 4):int(grid_size[2]/2 + 4)] = 2;
    
    # All matrices are internally transposed, therefore we need to transpose 
    # our source mask before computing indexes of the source points.
    source_mask = source_mask.transpose((2, 1, 0)).flatten()
    p_source_index = np.flatnonzero(source_mask)
    p_source_index = np.reshape(p_source_index, (p_source_index.size, 1, 1))
    
    FREQ = 1.0e6 # Hz
    AMPL = 50
    p_source_input_1 = np.reshape(AMPL * np.sin(FREQ * (2.0*math.pi) * np.arange(0.0, steps*dt, dt)), (1, steps, 1))
    
    FREQ = 2.0e6 # Hz
    AMPL = 100
    p_source_input_2 = np.reshape(AMPL * np.sin(FREQ * (2.0*math.pi) * np.arange(0.0, steps*dt, dt)), (1, steps, 1))
    
    p_source_input = np.zeros((p_source_index.size, steps, 1))
    p_source_input[source_mask[p_source_index.flat] == 1, :, :] = p_source_input_1
    p_source_input[source_mask[p_source_index.flat] == 2, :, :] = p_source_input_2
    
    kwave_file.set_p_source(p_source_index, p_source_input)

with h5py.File(args.filename, 'w') as dstDataFile:
    kwave_file.write_to_file(dstDataFile)
