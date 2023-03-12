from analyse import *
import os
import subprocess
from jinja2 import Template

proteins = initProteins()
proteins.to_csv('proteins.csv')

submission = Template("""#!/bin/bash
#PBS -W group_list=ku_10001 -A ku_10001
#PBS -N {{name}}
### Only send mail when job is aborted or terminates abnormally
#PBS -m n
### Number of nodes
#PBS -l nodes=1:ppn=2:gpus=1
### Memory
#PBS -e {{name}}
#PBS -o {{name}}
#PBS -l mem=10gb
#PBS -l walltime=61:00:00

source /home/people/giutes/.bashrc
conda activate cpeb4
module purge
module load cuda/toolkit/10.1/10.1.168 openmpi/gcc/64/4.0.2

cd $PBS_O_WORKDIR

python ./simulate.py --dirname {{name}} --proteins {{proteins}} --numbers {{numbers}} --temp {{temp}} --ionic {{ionic}} --cutoff {{cutoff}}""")

cutoff = 2.0

for name in ['CPEB4']:
    if not os.path.isdir(name):
        os.mkdir(name)
    for temp in [293]:
        if not os.path.isdir(name+'/{:d}'.format(temp)):
            os.mkdir(name+'/{:d}'.format(temp))
        for ionic in [60]:
            if not os.path.isdir(name+'/{:d}/{:d}'.format(temp,ionic)):
                os.mkdir(name+'/{:d}/{:d}'.format(temp,ionic))
            with open('{:s}_{:d}_{:d}.sh'.format(name,temp,ionic), 'w') as submit:
                submit.write(submission.render(name=name,proteins=name,ionic=ionic,
                              numbers='400',temp='{:d}'.format(temp),cutoff='{:.1f}'.format(cutoff)))
            subprocess.run(['qsub','{:s}_{:d}_{:d}.sh'.format(name,temp,ionic)])
