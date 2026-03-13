"""MNI152 standard brain surfaces.

Pre-extracted surfaces from the MNI152 T1 1mm brain using FSL BET.

Files
-----
- smri.nii.gz: MNI152 T1 structural MRI.
- inskull_mesh.nii.gz, inskull_mesh.vtk, inskull.png
- outskull_mesh.nii.gz, outskull_mesh.vtk, outskull.png
- outskin_mesh.nii.gz, outskin_mesh.vtk, outskin.png
- mni2mri_flirt_xform.txt: FLIRT transformation matrix (MNI to MRI).
- mni_mri-trans.fif: MNE transformation (MNI to MRI).
"""

from pathlib import Path

path = Path(__file__).parent
directory = str(path)
