from analyse import *
import os
import subprocess
from jinja2 import Template

proteins = initProteins()
proteins.to_pickle('proteins.pkl')

submission = Template("""#!/bin/bash
#PBS -W group_list=ku_10001 -A ku_10001
#PBS -N {{name}}_{{temp}}_map
#PBS -m n
#PBS -l nodes=1:ppn=1:thinnode
#PBS -l mem=10gb
#PBS -l walltime=10:00:00
#PBS -e {{name}}_{{temp}}_map_err
#PBS -o {{name}}_{{temp}}_map_out

source /home/people/giutes/.bashrc
conda activate cpeb4

cd $PBS_O_WORKDIR

python ./contact_maps.py --name {{name}} --temp {{temp}} --chunk {{chunk}}
""")

for name,prot in proteins.loc[['CPEB4','CPEB4pH6','CPEB4pH7']].iterrows():
    for temp in [293]:
        for chunk in range(10):
            with open('{:s}_{:d}_{:d}.sh'.format(name,temp,chunk), 'w') as submit:
                submit.write(submission.render(name=name,temp='{:d}'.format(temp),chunk='{:d}'.format(chunk)))
            subprocess.run(['qsub','{:s}_{:d}_{:d}.sh'.format(name,temp,chunk)])
