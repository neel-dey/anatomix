"""Exercise actual FireANTs stages, CLI batches, and label outputs."""
import csv
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import SimpleITK as sitk
import torch

from anatomix.registration.registration_infrastructure.cli import main, build_parser, build_stages
from anatomix.registration.registration_infrastructure.register import run_registration
from anatomix.registration.registration_infrastructure.warp_io import as_batch, warp_volume
from anatomix.registration.registration_infrastructure._fireants import Image, FakeBatchedImages
from anatomix.registration.tests.probe_linear import make_pair


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.device = os.environ.get('REGISTRATION_TEST_DEVICE','cpu')
        if cls.device.startswith('cuda'):
            torch.cuda.set_device(cls.device)

    def test_translation_recovery(self):
        f,m,points,targets = make_pair(None,'translation')
        fb,mb = as_batch(Image(f,device=self.device)),as_batch(Image(m,device=self.device))
        args = build_parser().parse_args(['--transform','rigid','--features','intensity','--loss','cc',
            '--shrink-factors','4x2x1','--iterations','100x100x100','--translation-step-size','0.01','--tolerance','inf'])
        result = run_registration(fb,mb,build_stages(args),
            reextract_moving=lambda grid: FakeBatchedImages(warp_volume(mb(),grid,'bilinear'),fb))
        matrix = result.stages[-1].linear_matrix.cpu().numpy()[0]
        pred = points @ matrix[:3,:3].T + matrix[:3,3]
        self.assertLess(np.linalg.norm(pred-targets,axis=1).mean(),.3)

    def test_cli_losses_chains_labels_and_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            z,y,x = np.mgrid[:36,:38,:40]
            arr = np.exp(-((x-19)**2+(y-18)**2+(z-17)**2)/80).astype('float32')
            for name, data in [('fixed',arr),('moving',arr),('mask',(arr>.01).astype('uint8')*255),('seg',(arr>.25).astype('int32')*40000)]:
                sitk.WriteImage(sitk.GetImageFromArray(data),str(root/f'{name}.nii.gz'))
            for loss in ['cc','mi','mse','masked_cc','masked_mi','masked_mse']:
                with self.subTest(loss=loss):
                    out = root/loss
                    main(['--fixed',str(root/'fixed.nii.gz'),'--moving',str(root/'moving.nii.gz'),
                          '--fixed-mask',str(root/'mask.nii.gz'),'--moving-mask',str(root/'mask.nii.gz'),
                          '--fixed-seg',str(root/'seg.nii.gz'),'--moving-seg',str(root/'seg.nii.gz'),
                          '--features','intensity','--loss',loss,'--step-size','0.1','--shrink-factors','1','--iterations','2',
                          '--output-dir',str(out),'--device',self.device,'--no-verbose'])
                    with (out/'metrics.csv').open() as f:
                        row = next(csv.DictReader(f))
                    self.assertGreater(float(row['dice']),.95)
                    self.assertTrue(np.isfinite(float(row['num_folds'])))
                    seg = sitk.GetArrayFromImage(sitk.ReadImage(str(out/'moved-seg-moving.nii.gz')))
                    self.assertEqual(set(np.unique(seg)),{0,40000})
            pairs = root/'pairs.csv'
            pairs.write_text('fixed,moving,subject\nfixed.nii.gz,moving.nii.gz,one\nfixed.nii.gz,moving.nii.gz,two\n')
            for initialization in ['none','image-centers','center-of-mass','moments']:
                with self.subTest(initialization=initialization):
                    out = root/initialization
                    main(['--registration-pairs-csv',str(pairs),'--features','mindssc',
                        '--transform','rigid,affine,deformable,deformable','--initialization',initialization,
                        '--shrink-factors','1,1,1,1','--iterations','0,0,0,0','--loss','mse,mse,mse,mse',
                        '--collapse-output-transforms','0','--output-transformation-convention','pytorch',
                        '--device',self.device,'--output-dir',str(out),'--no-verbose'])
                    with (out/'metrics.csv').open() as f:
                        rows = list(csv.DictReader(f))
                    self.assertEqual([r['subject'] for r in rows],['one','two'])
                    self.assertTrue((out/'moved-0-moving.nii.gz').exists())
                    self.assertTrue((out/'moved-1-moving.nii.gz').exists())
                    self.assertEqual(len(list(out.glob('warp-*.pt'))), 8 if initialization=='none' else 10)
                    if initialization == 'none':
                        moved = sitk.GetArrayFromImage(sitk.ReadImage(str(out/'moved-0-moving.nii.gz')))
                        np.testing.assert_allclose(moved,arr,atol=1e-5,rtol=1e-5)
            # Inverse export and an initial transform read back from an ANTs .mat.
            out = root/'inverse'
            main(['--fixed',str(root/'fixed.nii.gz'),'--moving',str(root/'moving.nii.gz'),
                  '--fixed-seg',str(root/'seg.nii.gz'),'--moving-seg',str(root/'seg.nii.gz'),
                  '--features','intensity','--transform','affine','--loss','mse','--iterations','0','--shrink-factors','1',
                  '--save-inverse','--device',self.device,'--output-dir',str(out),'--no-verbose'])
            with (out/'metrics.csv').open() as f:
                row = next(csv.DictReader(f))
            self.assertLess(float(row['inverse_residual_mm']), 1e-3)
            for name in ['warp-moving.mat','inverse-warp-moving.mat','inverse-moved-moving.nii.gz','inverse-moved-seg-moving.nii.gz']:
                self.assertTrue((out/name).exists(), name)
            np.testing.assert_allclose(sitk.GetArrayFromImage(sitk.ReadImage(str(out/'inverse-moved-moving.nii.gz'))),arr,atol=1e-5,rtol=1e-5)
            out2 = root/'from-mat'
            main(['--fixed',str(root/'fixed.nii.gz'),'--moving',str(root/'moving.nii.gz'),
                  '--features','intensity','--transform','deformable','--loss','mse','--iterations','0','--shrink-factors','1',
                  '--initial-transform',str(out/'warp-moving.mat'),'--collapse-output-transforms','0',
                  '--device',self.device,'--output-dir',str(out2),'--no-verbose'])
            self.assertTrue((out2/'warp-moving-init.mat').exists())
            self.assertTrue((out2/'warp-moving-0-deformable.nii.gz').exists())


if __name__ == '__main__':
    unittest.main()
