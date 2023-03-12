import os
import subprocess
from jinja2 import Template
import pandas as pd
import numpy as np

submission = Template("""#!/bin/bash
#PBS -W group_list=ku_10001 -A ku_10001
#PBS -N {{dirname}}_{{ionic}}
#PBS -m n
#PBS -l nodes=1:ppn=1:thinnode
#PBS -l mem=50gb
#PBS -l walltime=48:00:00
#PBS -e {{dirname}}_{{ionic}}_err
#PBS -o {{dirname}}_{{ionic}}_out

source /home/people/giutes/.bashrc
conda activate cpeb4

cd $PBS_O_WORKDIR

python ./ene_m.py --dirname {{dirname}} --ionic {{ionic}} --temp {{temp}} --size {{size}}""")

cutoff = 2.0

for name in ['CPEB4']:
    for temp in [293]:
        for ionic in [60]:
            for size in range(105,220,10):
                with open('{:s}_{:d}_{:d}.sh'.format(name,temp,ionic), 'w') as submit:
                    submit.write(submission.render(dirname=name,temp=temp,ionic=ionic,size=size))
                subprocess.run(['qsub','{:s}_{:d}_{:d}.sh'.format(name,temp,ionic)])
