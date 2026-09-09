"""Measure native-resolution recovery of known physical transforms."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import SimpleITK as sitk
import torch
from scipy.spatial.transform import Rotation

from anatomix.registration.registration_infrastructure.features import load_backbone, prepare_feature_channels, minmax_normalize
from anatomix.registration.registration_infrastructure.warp_io import as_batch
from anatomix.registration.registration_infrastructure._fireants import Image, RigidRegistration, AffineRegistration


def make_pair(source, transform, clip=True):
    if source:
        original = sitk.ReadImage(source, sitk.sitkFloat32)
        fixed = original
        arr = sitk.GetArrayFromImage(fixed)
        if clip:
            arr = np.clip(arr, -450, 450)
        normalized = sitk.GetImageFromArray(((arr - arr.min()) / (arr.max() - arr.min())).astype('float32'))
        normalized.CopyInformation(fixed)
        fixed = normalized
    else:
        z, y, x = np.mgrid[:48, :56, :64]
        arr = sum(a * np.exp(-((x-cx)**2/sx**2 + (y-cy)**2/sy**2 + (z-cz)**2/sz**2))
                  for a,cx,cy,cz,sx,sy,sz in [(1,24,22,20,9,12,7),(.7,43,35,32,6,8,5),(.5,18,39,29,4,5,6)])
        fixed = sitk.GetImageFromArray(arr.astype('float32'))
        fixed.SetSpacing((1.5, 2., 2.5))
        fixed.SetOrigin((-47., -55., -58.))
    center = fixed.TransformContinuousIndexToPhysicalPoint(tuple((n-1)/2 for n in fixed.GetSize()))
    matrix = np.eye(3) if transform == 'translation' else Rotation.from_euler('xyz', [4,-3,7], degrees=True).as_matrix()
    if transform == 'affine':
        matrix = matrix @ np.array([[1.04,.02,0],[0,.96,.01],[0,0,1.02]])
    known = sitk.AffineTransform(3)
    known.SetCenter(center)
    known.SetMatrix(matrix.ravel())
    known.SetTranslation((8., -5., 4.))
    moving = sitk.Resample(fixed, fixed, known.GetInverse(), sitk.sitkLinear, 0.)
    points = np.array([fixed.TransformContinuousIndexToPhysicalPoint(tuple(np.array(fixed.GetSize()) * f))
                       for f in ((.3,.3,.3),(.7,.3,.3),(.3,.7,.3),(.3,.3,.7),(.7,.7,.7),(.5,.5,.5))])
    targets = np.array([known.TransformPoint(p) for p in points])
    return fixed, moving, points, targets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source')
    p.add_argument('--features', default='intensity,anatomix-dev,anatomix-dev-vit')
    p.add_argument('--translation-multipliers', default='legacy,0.5,1,2')
    p.add_argument('--step-size', type=float, default=.01)
    p.add_argument('--transform', default='translation', choices=['translation','rigid','affine'])
    p.add_argument('--output', required=True)
    p.add_argument('--mask', action='store_true')
    p.add_argument('--cc-kernel', default='9x7x5')
    p.add_argument('--losses', default='cc')
    p.add_argument('--no-clip', action='store_true')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--iterations', type=int, default=100)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(12345)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)
    fixed, moving, points, targets = make_pair(args.source, args.transform, not args.no_clip)
    rows = []
    for family in args.features.split(','):
        f, m = Image(fixed, device=args.device), Image(moving, device=args.device)
        model = load_backbone(family, args.device) if family.startswith('anatomix') else None
        mode = 'anatomix' if model is not None else family
        for img in (f,m):
            mask = (img.array > .01).to(img.array.dtype)
            primary, mind = prepare_feature_channels(minmax_normalize(img.array), tuple(reversed(img.itk_image.GetSpacing())), model,
                features=mode, isotropic=True, window=128, sw_batch=4, overlap=.8, sw_mode='gaussian', sigma=.25,
                feature_normalization='l2', mindssc_radius=1, mindssc_dilation=2)
            img.array = primary if primary is not None else mind
            if args.mask:
                img.array = torch.cat([img.array*mask, mask],dim=1)
        del model
        fb, mb = as_batch(f), as_batch(m)
        for kind, cls in [('rigid',RigidRegistration), ('affine',AffineRegistration)]:
            for loss in args.losses.split(','):
                for multiplier in args.translation_multipliers.split(','):
                    normalized = multiplier != 'legacy'
                    translation_lr = args.step_size * float(multiplier) if normalized else None
                    start = time.monotonic()
                    reg = cls(scales=[4,2,1], iterations=[args.iterations]*3, fixed_images=fb, moving_images=mb,
                              loss_type=('masked_'+loss if args.mask else loss), cc_kernel_size=[int(x) for x in args.cc_kernel.split('x')], optimizer_lr=args.step_size, normalize_translation=normalized, translation_lr=translation_lr, progress_bar=False, tolerance=float('inf'))
                    reg.optimize()
                    matrix = (reg.get_rigid_matrix() if kind == 'rigid' else reg.get_affine_matrix()).detach().cpu().numpy()[0]
                    pred = points @ matrix[:3,:3].T + matrix[:3,3]
                    row = dict(source=args.source, shape_xyz=list(fixed.GetSize()), spacing_xyz=list(fixed.GetSpacing()),
                               known_transform=args.transform, masked=args.mask,
                               features=family, kind=kind, loss=loss, cc_kernel=args.cc_kernel,
                               step_size=args.step_size, translation_multiplier=multiplier,
                               translation_lr=translation_lr,
                               translation_radius=float(reg.translation_scale[0,0]),
                               iterations_per_level=args.iterations,
                               initial_tre=float(np.linalg.norm(points-targets,axis=1).mean()),
                               tre=float(np.linalg.norm(pred-targets,axis=1).mean()),
                               determinant=float(np.linalg.det(matrix[:3,:3])), seconds=time.monotonic()-start,
                               matrix=matrix.tolist())
                    rows.append(row)
                    print(json.dumps({k:v for k,v in row.items() if k != 'matrix'}), flush=True)
                    Path(args.output).write_text(json.dumps(rows, indent=2)+'\n')
                    del reg
        del f,m,fb,mb
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
