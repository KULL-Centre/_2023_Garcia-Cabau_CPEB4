from analyse import *
import os
import subprocess
import shutil
from jinja2 import Template

proteins = initProteins()
proteins.to_csv('proteins.csv')

submission = Template("""#!/bin/bash
#SBATCH --job-name={{name}}_{{replica}}
#SBATCH --nodes=1
#SBATCH --partition=qgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH -t 48:00:00
#SBATCH --mem=10GB
#SBATCH -o {{name}}/{{replica}}/out
#SBATCH -e {{name}}/{{replica}}/err

source /home/gitesei/.bashrc
module load gcc/11.2.0 openmpi/4.0.3 cuda/11.2.0
conda activate hyres

python ./simulate.py --name {{name}} --replica {{replica}}""")

for name,prot in proteins.loc[['CPEB4H50S_Clust','CPEB4H25S_Clust','CPEB4H50S','CPEB4H25S','CPEB4','CPEB4pH6','CPEB4pH7',]].iterrows():
    if not os.path.isdir(name):
        os.mkdir(name)
    for replica in [0,2,3]:
        with open(f'{name:s}_{replica:d}.sh', 'w') as submit:
            submit.write(submission.render(name=name,replica=f'{replica:d}'))
        subprocess.run(['sbatch',f'{name:s}_{replica:d}.sh'])
