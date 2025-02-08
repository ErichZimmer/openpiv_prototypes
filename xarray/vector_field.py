from __future__ import annotations

import numpy as np
import xarray as xr

from numbers import Number
from typing import Tuple, Union


UNITS_POS = 'px'
UNITS_VEL = 'dt'
UNITS_TIME = 'frame'
UNITS_ARB = 'n/a'
DIMS_2D = ['y', 'x']
DIMS_3D = ['z', 'y', 'x']
# Note: FLAGS constant should be removed
FLAGS = {
    0: 'valid',
    1: 'invalid',
    2: 'interpolated',
    3: 'masked'
}


class VectorField(object):
    def __init__(
        self, 
        x: np.ndarray=None,
        y: np.ndarray=None,
        u: np.ndarray=None,
        v: np.ndarray=None,
        s2n: np.ndarray=None,
        flag: np.ndarray=None,
        ds: xr.Dataset=None,
        frame: int = 0,
        units_pos: str=UNITS_POS,
        units_vel: str=UNITS_VEL,
        dtype='float64'
    ) -> None:
        self._ds = None
        units_vel = f'{units_pos}/{units_vel}'
        self.dtype = dtype

        # if the input data is numpy arrays, condition them into something more usable
        if (
             isinstance(x, np.ndarray) and
             isinstance(y, np.ndarray) and 
             isinstance(u, np.ndarray) and 
             isinstance(v, np.ndarray)
        ):
            # initialize defaults
            if not isinstance(s2n, np.ndarray):
                s2n = np.zeros_like(x)
    
            if not isinstance(flag, np.ndarray):
                flag = np.zeros_like(x)
            
            self._gen_from_ndarray(x, y, u, v, s2n, flag, frame)
            self._set_attrs(units_pos, units_vel)
            self._sanity_check()
        elif isinstance(ds, xr.Dataset):
            self._ds = ds
            self._sanity_check()
        else:
            raise TypeError(
                'Only numpy arrays and xarray Dataset are supported'
            )
            
    
    def _validate_grid_size(
        self,
        *kwargs
    ) -> None:
        num_tests = len(kwargs) - 1
        
        test = zip([kwargs[0]]*num_tests , kwargs[1:])
        if not all([a1.size == a2.size for a1, a2 in test]):
            raise ValueError(
                'Not all arrays are of the same size'
            )

    
    def _get_grid_shape(
        self, 
        x: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[int, int]:
        if len(x.shape) < 2:
            field_shape = (len(set(y), len(set(x))))
        else:
            field_shape = x.shape[:2]
            
        return field_shape


    def _gen_from_ndarray(
        self,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        s2n: np.ndarray,
        flag: np.ndarray,
        frame: int
    ) -> None:
        # validate grid size for each component
        self._validate_grid_size(x, y, u, v, s2n, flag)

        # condition data for dataset
        field_shape = self._get_grid_shape(x, y)
        
        x = np.unique(x)
        y = np.unique(y)
        
        u = u.reshape(field_shape)
        v = v.reshape(field_shape)
        s2n = s2n.reshape(field_shape)
        flag = flag.reshape(field_shape)

        x = x.astype(self.dtype)
        y = y.astype(self.dtype)
        u = u.astype(self.dtype)
        v = v.astype(self.dtype)
        s2n = s2n.astype(self.dtype)
        flag = flag.astype(self.dtype)

        # create xarray Dataset
        data = {
            'u': (DIMS_2D, u),
            'v': (DIMS_2D, v),
            's2n': (DIMS_2D, s2n),
            'flag': (DIMS_2D, flag),
        }

        coords = {
            'x': x,
            'y': y
        }
        
        ds = xr.Dataset(
            data,
            coords
        )

        # Add integer frame number
        ds = ds.expand_dims(
            dim={'t': [int(frame)]},
            axis=-1
        )

        self._ds = ds

        
    def _set_attrs(
        self, 
        units_pos: str, 
        units_vel: str
    ) -> None:        
        self._ds.attrs['dtype'] = str(self.dtype)
        
        self._ds.x.attrs['units'] = units_pos
        self._ds.y.attrs['units'] = units_pos
        self._ds.u.attrs['units'] = units_vel
        self._ds.v.attrs['units'] = units_vel
        self._ds.s2n.attrs['units'] = UNITS_ARB
        self._ds.flag.attrs['units'] = UNITS_ARB
        self._ds.t.attrs['units'] = UNITS_TIME

        
    # Note: we should probably check if sanity check actually works
    def _sanity_check(self) -> None:
        core_components = ['u', 'v', 'w', 's2n', 'flag']
        core_dims = ['x', 'y', 'z']

        ds = self.get_dataset()

        data_vars = [i for i in ds.data_vars]

        coords = []
        for data_name in data_vars:
            data = ds[data_name]
            data_coords = [data.coords[name].values for name in data.dims]
            coords.append(data_coords)

        indices = [(0, j) for j in range(1, len(coords))]
        for ind in indices:
            i, j = ind
            
            if (ds[data_vars[i]].dims != ds[data_vars[j]].dims):
                raise ValueError(
                    'All data vars must have the same dimension coordinates'
                )

            i_coords, j_coords = coords[i], coords[j]
            if not all([(ic == jc).all() for ic, jc in zip(i_coords, j_coords)]):
                raise ValueError(
                    'All data vars must have the same dimension coordinate values'
                )
            
    
    def get_dataset(self) -> xr.Dataset:
        return self._ds


    def get_meshgrid(self) -> np.ndarray:
        x, y = self._ds['x'], self._ds['y']
        x, y = np.meshgrid(x, y)

        return np.array([x, y], dtype=self.dtype)

    
    def get_u(self) -> xr.DataArray:
        return self._ds['u']

    
    def get_v(self) -> xr.DataArray:
        return self._ds['v']

    
    def get_s2n(self) -> xr.DataArray:
        return self._ds['s2n']

    
    def get_flag(self) -> xr.DataArray:
        return self._ds['flag']

    
    def get_subpixel_offset(self) -> Tuple[xr.DataArray, xr.DataArray]:
        u = self.get_u()
        v = self.get_v()

        u_px = u - u.astype(int)
        v_px = v - v.astype(int)

        u_px = abs(u_px)
        v_px = abs(v_px)

        return u_px, v_px

    
    def copy(
        self, 
        deep: bool=False
    ) -> VectorField:
        ds = self.get_dataset()
        
        return VectorField(ds=ds.copy(deep=deep))

    
    def extend(
        self,
        vectors: list[VectorField]
    ) -> VectorField:

        dataset = []

        # add all data to list to concat
        vectors.append(self)

        for data in vectors:
            dataset.append(data.get_dataset())
            
        new_ds = xr.concat(dataset, dim='t')

        # check for duplicate time instances, duplicates would cause issues later on
        t = new_ds.t.data

        if len(t) != len(np.unique(t)):
            raise ValueError(
                'Duplicate time instances are not valid'
            )

        # sort by time instance. Lowest time first.
        new_ds = new_ds.sortby('t')

        self._ds = new_ds


    ######################
    # FUNCTION OVERLOADS #
    ######################

    def __eq__(
        self, 
        rhs: VectorField
    ) -> bool:
        if not isinstance(rhs, VectorField):
            raise TypeError(
                f"Data type {str(type(rhs))} is not supported"
            )
                
        ds1 = self.get_dataset()
        ds2 = rhs.get_dataset()

        return ds1.equals(ds2)
        

    def __add__(
        self, 
        rhs: Union[Number, VectorField]
    ) -> VectorField:
        if isinstance(rhs, Number):
            self._ds['u'] = self._ds['u'] + rhs
            self._ds['v'] = self._ds['v'] + rhs
        elif isinstance(rhs, VectorField):
            self._ds['u'] = self._ds['u'] + rhs.ds['u']
            self._ds['v'] = self._ds['v'] + rhs.ds['v']
        else:
            raise TypeError(
                f"Data type {str(type(rhs))} is not supported"
            )
        
        return self

    
    def __sub__(
        self, 
        rhs: Union[Number, VectorField]
    ) -> VectorField:
        if isinstance(rhs, Number):
            self._ds['u'] = self._ds['u'] - rhs
            self._ds['v'] = self._ds['v'] - rhs
        elif isinstance(rhs, VectorField):
            self._ds['u'] = self._ds['u'] - rhs.ds['u']
            self._ds['v'] = self._ds['v'] - rhs.ds['v']
        else:
            raise TypeError(
                f"Data type {str(type(rhs))} is not supported"
            )
        
        return self

    
    def __mul__(
        self, 
        rhs: Union[Number, VectorField]
    ) -> VectorField:
        if isinstance(rhs, Number):
            self._ds['u'] =  self._ds['u'] * rhs
            self._ds['v'] =  self._ds['v'] * rhs
        elif isinstance(rhs, VectorField):
            self._ds['u'] = self._ds['u'] * rhs.ds['u']
            self._ds['v'] = self._ds['v'] * rhs.ds['v']
        else:
            raise TypeError(
                f"Data type {str(type(rhs))} is not supported"
            )
        
        return self

    
    def __truediv__(
        self, 
        rhs: Union[Number, VectorField]
    ) -> VectorField:
        if isinstance(rhs, Number):
            self._ds['u'] = self._ds['u'] / rhs
            self._ds['v'] = self._ds['v'] / rhs
        elif isinstance(rhs, VectorField):
            self._ds['u'] = self._ds['u'] / rhs.ds['u']
            self._ds['v'] = self._ds['v'] / rhs.ds['v']
        else:
            raise TypeError(
                f"Data type {str(type(rhs))} is not supported"
            )
        
        return self