# -*- coding: utf-8 -*-
"""Factory class for creating cumulative surface hopping trajectories"""

from .surface_hopping_md import SurfaceHoppingMD


class TrajectoryCum(SurfaceHoppingMD):
    """
    Factory class for creating SurfaceHoppingMD instances with cumulative hopping enabled.
    
    This class is maintained for backward compatibility. It simply creates a SurfaceHoppingMD
    instance with hopping_method="cumulative". All other options are passed through unchanged.
    """

    def __init__(self, *args, **kwargs):
        """Constructor that creates a SurfaceHoppingMD instance with cumulative hopping enabled"""
        kwargs['hopping_method'] = 'cumulative'
        super().__init__(*args, **kwargs) 