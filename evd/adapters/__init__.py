"""EVD integration adapters for video generator backbones."""

from .catlvdm_unet import CATLVDMEVDAdapter
from .stdit import STDiTEVDAdapter

__all__ = ["CATLVDMEVDAdapter", "STDiTEVDAdapter"]
