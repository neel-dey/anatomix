"""Registration contracts checked against physical coordinates and SimpleITK.

Run from the repository root: python -m unittest discover -s anatomix/registration/tests -v
"""
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from anatomix.registration.registration_infrastructure._fireants import Image, FakeBatchedImages
from anatomix.registration.registration_infrastructure import cli
from anatomix.registration.registration_infrastructure.features import combine_feature_channels
from anatomix.registration.registration_infrastructure.metrics import count_folds
from anatomix.registration.registration_infrastructure.register import run_registration, _linear_grid, _compose_linear_grid, _compose_grids
from anatomix.registration.registration_infrastructure.warp_io import (
    as_batch, _CumulativeWarp, _CumulativeLinear, warp_volume, warp_keypoints,
    keypoints_to_physical, keypoints_from_physical, invert_grid, read_linear_transform,
)


def image(size=(40, 38, 36), spacing=(1.2, 1.7, 2.1), reflected=False):
    z,y,x = np.mgrid[:size[2], :size[1], :size[0]]
    out = sitk.GetImageFromArray((x + 2*y + 3*z).astype('float32'))
    out.SetSpacing(spacing)
    out.SetOrigin((-21., -30., -36.))
    angle = .2
    direction = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    if reflected:
        direction[:,0] *= -1
    out.SetDirection(direction.ravel())
    return Image(out, device=os.environ.get('REGISTRATION_TEST_DEVICE','cpu'))


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.f = image()
        cls.m = image((44,42,39), spacing=(1.1,1.5,1.9))
        cls.fb, cls.mb = as_batch(cls.f), as_batch(cls.m)
        cls.matrix = torch.eye(4, device=cls.f.device)[None]
        cls.matrix[0,:3,3] = torch.tensor([1.2,-.7,.4], device=cls.f.device)
        cls.grid = _linear_grid(cls.matrix, cls.fb, cls.mb)

    def test_landmarks_all_conventions(self):
        vox = torch.tensor([[10.,12.,13.],[20.,18.,16.]], device=self.f.device)
        physical = keypoints_to_physical(vox,'voxel',self.f)
        expected = physical + self.matrix[0,:3,3]
        for convention in ['voxel','ras','lps']:
            with self.subTest(convention=convention):
                source = keypoints_from_physical(physical,convention,self.f)
                restored = keypoints_to_physical(source,convention,self.f)
                pred = warp_keypoints(restored,self.grid,self.f,self.m)
                torch.testing.assert_close(pred, expected, atol=2e-5, rtol=1e-5)

    def test_image_centers_initialization(self):
        moving_itk = sitk.Image(image((44,42,39), spacing=(1.1,1.5,1.9), reflected=True).itk_image)
        moving_itk.SetOrigin((17., -9., 25.))
        moving = Image(moving_itk, device=self.f.device)
        fixed_center = self.f.itk_image.TransformContinuousIndexToPhysicalPoint(
            tuple((np.array(self.f.itk_image.GetSize())-1)/2))
        moving_center = moving_itk.TransformContinuousIndexToPhysicalPoint(
            tuple((np.array(moving_itk.GetSize())-1)/2))
        translation = torch.tensor(np.array(moving_center)-fixed_center,
                                   device=self.f.device, dtype=torch.float32)
        # Signed, zero-sum fixed intensities would make intensity COM undefined.
        fixed = FakeBatchedImages(self.fb()-self.fb().mean(), self.fb)
        mb = as_batch(moving)
        moving_batch = FakeBatchedImages(mb().abs()+1, mb)
        points = torch.tensor(np.array([fixed_center, np.array(fixed_center)+[3.,-5.,7.]]),
                              device=self.f.device, dtype=torch.float32)
        for chain in ['rigid', 'affine', 'deformable', 'rigid,affine,deformable']:
            with self.subTest(chain=chain):
                n = len(chain.split(','))
                args = cli.build_parser().parse_args([
                    '--initialization', 'image-centers', '--transform', chain,
                    '--iterations', ','.join(['0']*n),
                    '--shrink-factors', ','.join(['1']*n),
                    '--loss', ','.join(['mse']*n)])
                result = run_registration(
                    fixed, moving_batch, cli.build_stages(args),
                    initialization=args.initialization,
                    reextract_moving=lambda grid: fixed)
                for _, grid in result.snapshots:
                    pred = warp_keypoints(points, grid, self.f, moving)
                    torch.testing.assert_close(pred, points+translation, atol=3e-5, rtol=1e-5)

    def test_center_of_mass_with_signed_intensities(self):
        # A CT-like image with negative background: shifting the moving image's
        # origin must be recovered by the intensity center of mass.
        signed = self.fb() * 2000 - 1000
        moving_itk = sitk.Image(self.f.itk_image)
        moving_itk.SetOrigin(tuple(np.array(moving_itk.GetOrigin()) + [6., -4., 9.]))
        moving = as_batch(Image(moving_itk, device=self.f.device))
        from anatomix.registration.registration_infrastructure.features import minmax_normalize
        mass = minmax_normalize(signed, -450, 450)
        init = (FakeBatchedImages(mass, self.fb), FakeBatchedImages(mass, moving))
        args = cli.build_parser().parse_args(['--transform','rigid','--iterations','0','--shrink-factors','1','--loss','mse'])
        result = run_registration(self.fb, moving, cli.build_stages(args), initialization='center-of-mass',
                                  init_images=init, reextract_moving=lambda grid: self.fb)
        matrix = result.stages[-1].linear_matrix[0]
        expected = torch.tensor([6., -4., 9.], device=matrix.device)  # ITK origins are LPS
        torch.testing.assert_close(matrix[:3, 3], expected, atol=2e-3, rtol=0)
        torch.testing.assert_close(matrix[:3, :3], torch.eye(3, device=matrix.device), atol=1e-6, rtol=0)

    def test_linear_composition_outside_fov(self):
        residual = F.affine_grid(torch.eye(3,4,device=self.f.device)[None],self.fb.shape,align_corners=True)
        residual = residual + .4
        result = _compose_linear_grid(self.matrix,residual,self.fb,self.mb)
        physical = residual @ self.fb.get_torch2phy()[0,:3,:3].T + self.fb.get_torch2phy()[0,:3,3]
        physical = physical @ self.matrix[0,:3,:3].T + self.matrix[0,:3,3]
        expected = physical @ self.mb.get_phy2torch()[0,:3,:3].T + self.mb.get_phy2torch()[0,:3,3]
        torch.testing.assert_close(result,expected,atol=2e-6,rtol=1e-5)

    def test_dense_composition_matches_analytic_interior(self):
        identity = F.affine_grid(torch.eye(3,4,device=self.f.device)[None],self.fb.shape,align_corners=True)
        old = identity.clone()
        old[...,0] += .1*identity[...,1]*identity[...,2]
        residual = identity*.8
        residual[...,1] += .03
        expected = residual.clone()
        expected[...,0] += .1*residual[...,1]*residual[...,2]
        torch.testing.assert_close(_compose_grids(old,residual),expected,atol=2e-6,rtol=1e-5)

    def test_ants_dense_and_linear_exports(self):
        grid = self.grid.clone()
        grid[...,0] += .02*torch.sin(grid[...,1]*3)*torch.cos(grid[...,2]*2)
        for dense in [False,True]:
            with self.subTest(dense=dense), tempfile.TemporaryDirectory() as tmp:
                path = str(Path(tmp)/('warp.nii.gz' if dense else 'warp.mat'))
                if dense:
                    _CumulativeWarp(grid,self.fb,self.mb).save_as_ants_transforms(path)
                    field = sitk.ReadImage(path,sitk.sitkVectorFloat64)
                    transform = sitk.DisplacementFieldTransform(field)
                else:
                    _CumulativeLinear(self.matrix).save_as_ants_transforms(path)
                    transform = sitk.ReadTransform(path)
                reference = sitk.Resample(self.m.itk_image,self.f.itk_image,transform,sitk.sitkLinear,0.)
                result = warp_volume(self.m.array,grid if dense else self.grid,'bilinear').cpu().numpy()[0,0]
                # ITK and torch use different padding at the half-voxel boundary.
                query = (grid if dense else self.grid).cpu().numpy()[0]
                interior = np.all(np.abs(query)<.95,axis=-1)
                np.testing.assert_allclose(result[interior],sitk.GetArrayFromImage(reference)[interior],atol=1e-4,rtol=1e-5)

    def test_scipy_export_unequal_grids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/'warp.npz')
            _CumulativeWarp(self.grid,self.fb,self.mb).save_as_scipy_transforms(path)
            disp = np.load(path)['arr_0']
            axes = np.moveaxis(np.indices(self.f.itk_image.GetSize()),0,-1)
            sample = axes + disp
            expected = (self.grid.cpu().numpy()[0].transpose(2,1,0,3)+1) * (np.array(self.m.itk_image.GetSize())-1)/2
            np.testing.assert_allclose(sample,expected,atol=.025,rtol=0)

    def test_physical_identity_has_no_folds_with_reflected_header(self):
        reflected = as_batch(image(reflected=True))
        grid = _linear_grid(torch.eye(4,device=self.f.device)[None],self.fb,reflected)
        self.assertEqual(count_folds(grid),int(np.prod(np.array(self.fb.shape[2:])-2)))
        self.assertEqual(count_folds(grid,self.fb,reflected),0)

    def test_inverse_of_linear_grid_is_inverse_matrix(self):
        grid_inv, residual = invert_grid(self.grid, self.fb, self.mb)
        expected = _linear_grid(torch.linalg.inv(self.matrix), self.mb, self.fb)
        torch.testing.assert_close(grid_inv, expected, atol=1e-4, rtol=1e-5)
        self.assertLess(float(residual[torch.isfinite(residual)].max()), 1e-3)

    def test_inverse_of_dense_grid_round_trips(self):
        grid = self.grid.clone()
        grid[...,0] += .03*torch.sin(grid[...,1]*3)*torch.cos(grid[...,2]*2)
        grid_inv, residual = invert_grid(grid, self.fb, self.mb)
        finite = residual[torch.isfinite(residual)]
        self.assertGreater(finite.numel(), residual.numel()//2)
        self.assertLess(float(finite.max()), 1e-2)
        # A moving-grid point mapped to fixed space and back returns to the same point.
        fixed_pts = keypoints_to_physical(torch.tensor([[12.,14.,11.],[30.,20.,19.]], device=self.f.device),'voxel',self.f)
        moving_pts = warp_keypoints(fixed_pts, grid, self.f, self.m)
        back = warp_keypoints(moving_pts, grid_inv, self.m, self.f)
        torch.testing.assert_close(back, fixed_pts, atol=2e-2, rtol=0)

    def test_initial_transform_file_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/'init.mat')
            _CumulativeLinear(self.matrix).save_as_ants_transforms(path)
            matrix = read_linear_transform(path)
            torch.testing.assert_close(matrix, self.matrix.cpu(), atol=1e-5, rtol=1e-5)
            args = cli.build_parser().parse_args(['--transform','rigid','--iterations','0','--shrink-factors','1','--loss','mse'])
            result = run_registration(self.fb, self.mb, cli.build_stages(args),
                                      initial_transform=matrix, reextract_moving=lambda grid: self.fb)
            torch.testing.assert_close(result.stages[-1].linear_matrix.cpu(), self.matrix.cpu(), atol=1e-5, rtol=1e-5)
            self.assertEqual([name for name, _ in result.snapshots], ['init', '0-rigid'])
            with self.assertRaisesRegex(ValueError, 'initialization'):
                run_registration(self.fb, self.mb, cli.build_stages(args), initialization='image-centers',
                                 initial_transform=matrix, reextract_moving=lambda grid: self.fb)

    def test_lta_transform_matches_itk(self):
        # A FreeSurfer LTA is src->dst in RAS. With src = moving and dst = fixed it
        # is the inverse of the ITK fixed->moving matrix; vox-to-vox files convert
        # through the stored volume geometry.
        def geometry(image):
            itk = image.itk_image; shape = np.array(itk.GetSize(), dtype=float)
            direction = np.array(itk.GetDirection()).reshape(3,3); spacing = np.array(itk.GetSpacing())
            affine = np.eye(4); affine[:3,:3] = direction * spacing; affine[:3,3] = itk.GetOrigin()
            affine = np.diag([-1,-1,1,1]) @ affine  # LPS -> RAS
            return shape, affine
        def block(name, shape, affine):
            axes = affine[:3,:3] / np.linalg.norm(affine[:3,:3], axis=0); vs = np.linalg.norm(affine[:3,:3], axis=0)
            cras = affine[:3,:3] @ (shape/2.0) + affine[:3,3]
            return (f"{name} volume info\nvalid = 1\nfilename = x\nvolume = {' '.join(str(int(v)) for v in shape)}\n"
                    f"voxelsize = {' '.join(map(str, vs))}\nxras = {' '.join(map(str, axes[:,0]))}\n"
                    f"yras = {' '.join(map(str, axes[:,1]))}\nzras = {' '.join(map(str, axes[:,2]))}\ncras = {' '.join(map(str, cras))}\n")
        fixed, moving = geometry(self.f), geometry(self.m)
        lps = self.matrix[0].cpu().numpy().astype(float); D = np.diag([-1.,-1.,1.,1.])
        src_to_dst = np.linalg.inv(D @ lps @ D)  # moving RAS -> fixed RAS
        def lta_text(kind, matrix):
            rows = '\n'.join(' '.join(f'{float(v):.17g}' for v in row) for row in matrix)
            return f"type = {kind}\nnxforms = 1\nmean = 0 0 0\nsigma = 1\n1 4 4\n{rows}\n" + block('src', *moving) + block('dst', *fixed)
        with tempfile.TemporaryDirectory() as tmp:
            ras = Path(tmp)/'ras.lta'; ras.write_text(lta_text(1, src_to_dst))
            torch.testing.assert_close(read_linear_transform(str(ras), fixed, moving), self.matrix.cpu(), atol=1e-4, rtol=1e-5)
            vox = Path(tmp)/'vox.lta'
            vox.write_text(lta_text(0, np.linalg.inv(fixed[1]) @ src_to_dst @ moving[1]))
            torch.testing.assert_close(read_linear_transform(str(vox), fixed, moving), self.matrix.cpu(), atol=1e-4, rtol=1e-5)
            # src = fixed, dst = moving is detected from the geometry and not inverted
            swapped = Path(tmp)/'swapped.lta'
            swapped.write_text(f"type = 1\nnxforms = 1\nmean = 0 0 0\nsigma = 1\n1 4 4\n" +
                '\n'.join(' '.join(f'{float(v):.17g}' for v in row) for row in (D @ lps @ D)) + '\n' + block('src', *fixed) + block('dst', *moving))
            torch.testing.assert_close(read_linear_transform(str(swapped), fixed, moving), self.matrix.cpu(), atol=1e-4, rtol=1e-5)

    def test_physical_reflection_has_folds(self):
        mat = self.matrix.clone()
        mat[:,0,0] = -1
        grid = _linear_grid(mat,self.fb,self.mb)
        expected = np.prod(np.array(self.fb.shape[2:])-2)
        self.assertEqual(count_folds(grid,self.fb,self.mb), expected)


class InputTests(unittest.TestCase):
    def test_mask_values_are_foreground_not_weights(self):
        primary = torch.ones(1,2,3,3,3)
        mask = torch.ones(1,1,3,3,3)*255
        torch.testing.assert_close(combine_feature_channels(primary,None,mask,'anatomix'),primary)

    def test_nonfinite_keypoints_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'points.csv'
            path.write_text('x,y,z\nnan,1,2\n')
            with self.assertRaisesRegex(ValueError,'finite'):
                cli.validate_keypoints([{'fixed_keypoints':str(path)}])

    def test_aux_geometry_mismatch_rejected(self):
        geom = {'fixed':((40,40,40),np.eye(4)), 'fixed_mask':((40,40,40),np.diag([2,1,1,1]))}
        with self.assertRaisesRegex(ValueError,'affine differs'):
            cli.validate_geometry([geom])

    def test_initial_transform_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bad.mat'; path.write_bytes(b'not a transform')
            with self.assertRaisesRegex(ValueError, 'initial_transform'):
                cli.validate_initial_transforms([{'initial_transform': str(path)}], [{}], 'none')
            good = Path(tmp)/'good.mat'
            _CumulativeLinear(torch.eye(4)[None]).save_as_ants_transforms(str(good))
            geometry = {'fixed': ((40,38,36), np.eye(4)), 'moving': ((44,42,39), np.eye(4))}
            cli.validate_initial_transforms([{'initial_transform': str(good)}], [geometry], 'none')
            with self.assertRaisesRegex(ValueError, 'cannot be combined'):
                cli.validate_initial_transforms([{'initial_transform': str(good)}], [geometry], 'center-of-mass')

    def test_schedule_rejects_wrong_stage_count(self):
        args = cli.build_parser().parse_args(['--transform','rigid,affine','--iterations','100'])
        with self.assertRaises(ValueError):
            cli.build_stages(args)


if __name__ == '__main__':
    unittest.main()
