"""Run a small, explicitly selected manifest through full CLI recipes."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--anatomy', choices=['brain','abdomen'], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--limit', type=int, default=3)
    args = p.parse_args()
    manifest = Path(args.manifest).resolve()
    with manifest.open() as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        rows = list(reader)[:args.limit]
    path_columns = {'fixed','moving','fixed_mask','moving_mask','fixed_seg','moving_seg','fixed_keypoints','moving_keypoints'}
    for row in rows:
        for key in path_columns & row.keys():
            if row[key] and not Path(row[key]).is_absolute():
                row[key] = str(manifest.parent / row[key])
    shared = ['--no-verbose','--backbone','anatomix-dev-vit','--device','cuda:0','--sliding-window-params','128,4,0.8,gaussian,0.25']
    if args.anatomy == 'brain':
        recipes = {
            'deformable-conservative': ['--transform','deformable','--step-size','0.1','--shrink-factors','4x2x1','--iterations','200x100x50','--cc-kernel-widths','7x5x3'],
            'affine-features': ['--transform','affine','--step-size','0.01','--translation-step-size','0.01','--shrink-factors','4x2x1','--iterations','100x100x100','--cc-kernel-widths','9x7x5'],
            'affine-deformable': ['--transform','affine,deformable','--step-size','0.01,0.1','--translation-step-size','0.01,na','--shrink-factors','4x2x1,4x2x1','--iterations','100x100x100,200x100x50','--cc-kernel-widths','9x7x5,7x5x3'],
        }
    else:
        shared += ['--fixed-minclip','-450','--fixed-maxclip','450','--moving-minclip','0','--moving-maxclip','20000']
        recipes = {'deformable-wide-cc': ['--step-size','1.0','--shrink-factors','6x4x2x1','--iterations','100x100x100x100','--cc-kernel-widths','21x13x11x9']}
    results = []
    for name, flags in recipes.items():
        with tempfile.TemporaryDirectory(prefix='recipe-') as tmp:
            pair_file = Path(tmp)/'pairs.csv'
            with pair_file.open('w') as f:
                writer=csv.DictWriter(f,fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            start = time.monotonic()
            cmd = [sys.executable,'-m','anatomix.registration.registration_infrastructure.cli','--registration-pairs-csv',str(pair_file),'--output-dir',tmp] + shared + flags
            subprocess.run(cmd,check=True)
            with (Path(tmp)/'metrics.csv').open() as f:
                metrics=list(csv.DictReader(f))
            result=dict(recipe=name, flags=shared+flags, seconds=time.monotonic()-start, metrics=metrics)
            results.append(result)
            Path(args.output).write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
