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
#PBS -l walltime=50:00:00

source /home/people/giutes/.bashrc
conda activate cpeb4
module purge
module load cuda/toolkit/10.1/10.1.168 openmpi/gcc/64/4.0.2

cd $PBS_O_WORKDIR

python ./simulate.py --name {{name}} --temp {{temp}} --cutoff {{cutoff}}""")

cutoff = 2.0

for name,prot in proteins.loc[['CPEB4','CPEB4pH6','CPEB4pH7','CPEB4H50S','CPEB4H50S_Clust','CPEB4H25S','CPEB4H25S_Clust']].iterrows():
    if not os.path.isdir(name):
        os.mkdir(name)
    for temp in [293]:
        if not os.path.isdir(f'{name:s}/{temp:d}'):
            os.mkdir(f'{name:s}/{temp:d}')
        with open(f'{name:s}_{temp:d}.sh', 'w') as submit:
            submit.write(submission.render(name=name,temp='{:d}'.format(temp),cutoff='{:.1f}'.format(cutoff)))
        subprocess.run(['qsub',f'{name:s}_{temp:d}.sh'])
