"""
 * @file      kwave_data_filters.py
 *
 * @author    Filip Vaverka
 *            Faculty of Information Technology
 *            Brno University of Technology
 *            ivaverka@fit.vutbr.cz
 *
 * @brief     Windowing of functions using spectral approach.
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

import numpy as np
from scipy.signal import blackman
import math


def window_filter(data, filter_func):
    for axis, axis_size in enumerate(data.shape):
        filter_shape = [1, ] * data.ndim

        filter_shape[axis] = axis_size
        window_size = (2 * axis_size - 1) if axis == 2 else axis_size

        w = filter_func(window_size) + np.finfo(np.float32).eps
        w = np.fft.ifftshift(w)
        w = w[0:axis_size].reshape(filter_shape)
        data *= np.power(w, 1.0 / data.ndim)


class SpectralDataFilter(object):
    @staticmethod
    def smooth(data, window='blackman'):
        data_fe = np.fft.rfftn(data, data.shape)

        if window == 'blackman':
            window_func = blackman
        else:
            raise ValueError('Unsupported spectrum windowing function')

        window_filter(data_fe, window_func)

        return np.fft.irfftn(data_fe, data.shape)
