from analyse import *
import os
import subprocess
from jinja2 import Template
import time

proteins = initProteins()
proteins.to_pickle('proteins.pkl')

submission = Template("""#!/bin/bash
#SBATCH --job-name={{name}}
#SBATCH --nodes=1
#SBATCH --partition=qgpu
#SBATCH --cpus-per-task=1
#SBATCH -t 10:00:00
#SBATCH --mem=20GB
#SBATCH -o {{name}}/cmap_{{replica}}_out
#SBATCH -e {{name}}/cmap_{{replica}}_err

source /home/gitesei/.bashrc
module load cmake/3.9.4 gcc/6.5.0 openmpi/4.0.3 llvm/7.0.0 cuda/9.2.148 eigen/3.3.2
conda activate openmm

python ./contact_maps.py --name {{name}} --temp {{temp}} --replica {{replica}} --chunk {{chunk}}
""")

temp = 293

if not os.path.isdir('contact_maps'):
    os.mkdir('contact_maps')
for name,prot in proteins.loc[['CPEB4','CPEB4pH6']].iterrows():
    for replica in [0,2,3]:
        for chunk in range(10):
            with open('{:s}_{:d}_{:d}.sh'.format(name,replica,chunk), 'w') as submit:
                submit.write(submission.render(name=name,temp='{:d}'.format(temp),replica='{:d}'.format(replica),chunk='{:d}'.format(chunk)))
            subprocess.run(['sbatch','{:s}_{:d}_{:d}.sh'.format(name,replica,chunk)])
            time.sleep(0.5)
