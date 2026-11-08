"""unroller: virtual unrolling of scroll segments.

Pipeline:  Volume (OME-Zarr)  +  Surface (tifxyz)
             -> flatten (LSCM init + ARAP isometric refinement)
             -> resample to a flat tifxyz
             -> render a layered stack along surface normals
             -> metrics (distortion, sheet centering, ink fidelity on phantoms)
"""
from .volume import Volume
from .surface import Surface

__all__ = ["Volume", "Surface"]
__version__ = "0.2.0"
