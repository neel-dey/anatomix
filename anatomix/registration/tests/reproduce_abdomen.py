"""Reproduce the eight-pair AbdomenMRCT configuration at native resolution."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--cases', default='1,2,3,4,5,6,7,8')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows=[]
    for case in map(int,args.cases.split(',')):
        row={'case':f'{case:04d}'}
        for role, modality in [('fixed','0001'),('moving','0000')]:
            for folder,suffix in [('imagesTr',''),('masksTr','_mask'),('labelsTr','_seg')]:
                path=args.data_root/folder/f'AbdomenMRCT_{case:04d}_{modality}.nii.gz'
                if not path.is_file(): raise FileNotFoundError(path)
                row[role+suffix]=str(path.resolve())
        rows.append(row)
    manifest=args.output_dir/'pairs.csv'
    with manifest.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    flags=['--registration-pairs-csv',str(manifest),'--output-dir',str(args.output_dir),
        '--features','anatomix+mindssc','--backbone','anatomix-dev-vit',
        '--transform','deformable','--loss','masked_cc','--initialization','none',
        '--step-size','1.0','--shrink-factors','6x4x2x1','--iterations','100x100x100x100',
        '--cc-kernel-widths','21x13x11x9','--smooth-grad-sigma','1.0','--smooth-warp-sigma','0.5',
        '--fixed-minclip','-450','--fixed-maxclip','450','--moving-minclip','0','--moving-maxclip','20000',
        '--isotropic-features','1','--feature-normalization','l2',
        '--sliding-window-params','128,4,0.8,gaussian,0.25','--mindssc-params','1,2',
        '--seed','12345','--tolerance','1e-6','--device',args.device,'--no-verbose']
    (args.output_dir/'command.json').write_text(json.dumps(flags,indent=2)+'\n')
    subprocess.run([sys.executable,'-m','anatomix.registration.registration_infrastructure.cli']+flags,check=True)
    with (args.output_dir/'metrics.csv').open() as f: metrics=list(csv.DictReader(f))
    summary={'mean_dice':sum(float(r['dice']) for r in metrics)/len(metrics),
             'total_folds':sum(int(r['num_folds']) for r in metrics),'metrics':metrics}
    if sorted(int(row['case']) for row in metrics) == list(range(1,9)):
        summary['reference_mean_dice'] = 0.879067
        summary['mean_dice_tolerance'] = 0.001
        summary['accepted'] = abs(summary['mean_dice'] - 0.879067) <= 0.001 and summary['total_folds'] == 0
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if summary.get('accepted') is False:
        raise RuntimeError("AbdomenMRCT result differs from the recorded Dice/fold acceptance target")
    print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
