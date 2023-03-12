from analyse import *
import subprocess
from jinja2 import Template

submission = Template("""#!/bin/bash
#PBS -W group_list=ku_10001 -A ku_10001
#PBS -N {{dirname}}_cluster
#PBS -m n
#PBS -l nodes=1:ppn=1:thinnode
#PBS -l mem=20gb
#PBS -l walltime=2:00:00
#PBS -e {{dirname}}_cluster_err
#PBS -o {{dirname}}_cluster_out

source /home/people/giutes/.bashrc
conda activate cpeb4

cd $PBS_O_WORKDIR

python ./cm_trajectory.py --dirname {{dirname}} --proteins {{proteins}} --numbers {{numbers}} --temp {{temp}} --ionic {{ionic}}
python ./cluster_analysis.py --dirname {{dirname}} --temp {{temp}} --ionic {{ionic}}""")

for name in ['CPEB4']:
    for temp in [293]:
        for ionic in [60]:
            with open('{:s}_{:d}_{:d}.sh'.format(name,temp,ionic), 'w') as submit:
                submit.write(submission.render(dirname=name,proteins=name,ionic=ionic,
                          numbers='400',temp=temp))
            subprocess.run(['qsub','{:s}_{:d}_{:d}.sh'.format(name,temp,ionic)])
