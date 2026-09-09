"""Public CLI contracts for one-sided masks and model-free linear registration."""
import csv
from itertools import product
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk
import torch

from anatomix.registration.registration_infrastructure import cli
from anatomix.registration.registration_infrastructure.pipeline import resolve_stage_losses


class MaskAndIntensityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.device=os.environ.get('REGISTRATION_TEST_DEVICE','cpu')
        if cls.device.startswith('cuda'): torch.cuda.set_device(cls.device)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        for role,shape in [('fixed',(36,38,40)),('moving',(38,40,42))]:
            coords=np.indices(shape)
            arr=np.exp(-sum((coords[i]-shape[i]*.5)**2 for i in range(3))/70).astype('float32')
            for folder,data in [(role,arr),(role+'_mask',(arr>.03).astype('uint8')*255),(role+'_ones',np.ones(shape,dtype='uint8'))]:
                directory=self.root/folder;directory.mkdir()
                image=sitk.GetImageFromArray(data);image.SetSpacing((1.2,1.7,2.))
                sitk.WriteImage(image,str(directory/'image.nii.gz'))

    def tearDown(self): self.tmp.cleanup()

    def path(self,role): return str(self.root/role/'image.nii.gz')

    def test_one_sided_masks_match_explicit_ones_through_chain(self):
        for side, losses in product(['fixed','moving'], [None,'cc,masked_mse,masked_cc']):
            with self.subTest(side=side,losses=losses):
                other='moving' if side=='fixed' else 'fixed'
                flags=['--fixed',self.path('fixed'),'--moving',self.path('moving'),
                    '--'+side+'-mask',self.path(side+'_mask'), '--features','intensity',
                    '--transform','rigid,affine,deformable','--step-size','0.001,0.001,0.1',
                    '--shrink-factors','1,1,1','--iterations','2,2,2',
                    '--output-transformation-convention','pytorch','--device',self.device,'--no-verbose']
                if losses:
                    flags += ['--loss',losses]
                outputs=[]
                for explicit in [False,True]:
                    out=self.root/f'{side}-{explicit}'
                    args=flags+['--output-dir',str(out)]
                    if explicit: args+=['--'+other+'-mask',self.path(other+'_ones')]
                    with patch('anatomix.registration.registration_infrastructure.pipeline.load_backbone',side_effect=AssertionError('intensity must not load a backbone')):
                        cli.main(args)
                    outputs.append(torch.load(out/'warp-image.pt',weights_only=True))
                torch.testing.assert_close(*outputs,atol=2e-6,rtol=2e-6)

    def test_pair_csv_and_directory_resolve_either_mask(self):
        for side in ['fixed','moving']:
            with self.subTest(side=side):
                pair_flags=['--fixed',self.path('fixed'),'--moving',self.path('moving'),'--'+side+'-mask',self.path(side+'_mask')]
                dir_flags=['--fixed-dir',str(self.root/'fixed'),'--moving-dir',str(self.root/'moving'),'--'+side+'-mask-dir',str(self.root/(side+'_mask'))]
                manifest=self.root/f'{side}.csv'
                manifest.write_text(f'fixed,moving,{side}_mask\nfixed/image.nii.gz,moving/image.nii.gz,{side}_mask/image.nii.gz\n')
                for flags in [pair_flags,dir_flags,['--registration-pairs-csv',str(manifest)]]:
                    args=cli.build_parser().parse_args(flags)
                    pairs,_=cli.resolve_inputs(args);stages=cli.build_stages(args)
                    cli.validate_pairs(pairs,stages)
                    self.assertEqual(pairs[0][side+'_mask'],self.path(side+'_mask'))
                    self.assertEqual(resolve_stage_losses(stages,True)[0]['loss'],'masked_cc')
        self.assertEqual(resolve_stage_losses([{'loss':None}],False)[0]['loss'],'cc')
        self.assertEqual(resolve_stage_losses([{'loss':'mse'}],True)[0]['loss'],'mse')

    def test_intensity_only_rigid_and_affine_cli(self):
        for kind in ['rigid','affine']:
            for loss in ['cc','mi','mse']:
                with self.subTest(kind=kind,loss=loss):
                    out=self.root/f'{kind}-{loss}'
                    with patch('anatomix.registration.registration_infrastructure.pipeline.load_backbone',side_effect=AssertionError('no backbone expected')):
                        cli.main(['--fixed',self.path('fixed'),'--moving',self.path('moving'),
                            '--features','intensity','--transform',kind,'--loss',loss,
                            '--step-size','0.001','--shrink-factors','1','--iterations','2',
                            '--output-dir',str(out),'--device',self.device,'--no-verbose'])
                    self.assertTrue((out/'warp-image.mat').is_file())
                    with (out/'metrics.csv').open() as f: row=next(csv.DictReader(f))
                    self.assertEqual(int(row['num_folds']),0)


if __name__=='__main__': unittest.main()
