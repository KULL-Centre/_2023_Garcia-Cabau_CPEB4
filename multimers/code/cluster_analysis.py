from ovito.io import import_file, export_file
from ovito.modifiers import LoadTrajectoryModifier
from ovito.modifiers import ClusterAnalysisModifier
from ovito.modifiers import UnwrapTrajectoriesModifier
from ovito.modifiers import CalculateDisplacementsModifier
from ovito.modifiers import SelectTypeModifier
from ovito.modifiers import CoordinationAnalysisModifier
import numpy as np
import mdtraj as md
import itertools
import pandas as pd

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--dirname',nargs='?',const='',type=str,required=True)
parser.add_argument('--temp',nargs='?',const='',type=int,required=True)
parser.add_argument('--ionic',nargs='?',const='',type=int,required=True)
args = parser.parse_args()

def calc_kappa(tensor):
    xx = tensor[0]
    yy = tensor[1]
    zz = tensor[2]
    return 3/2*(xx**2+yy**2+zz**2)/(xx+yy+zz)**2-1/2

def cluster_analysis(dirname,N,temp,ionic,rg,cluster,num_particles,particle_types={'B','C'}):
    pipeline = import_file('cmtraj/{:s}_400_{:d}_{:d}.pdb'.format(dirname,temp,ionic))
    traj_mod = LoadTrajectoryModifier()
    traj_mod.source.load('cmtraj/{:s}_400_{:d}_{:d}.xtc'.format(dirname,temp,ionic))
    print("Number of frames: ", traj_mod.source.num_frames)
    pipeline.modifiers.append(traj_mod)
    pipeline.modifiers.append(SelectTypeModifier(property = 'Particle Type', types = particle_types))
    pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=rg*15,only_selected=True))
    pipeline.modifiers.append(ClusterAnalysisModifier(
        cutoff=rg*15, neighbor_mode=ClusterAnalysisModifier.NeighborMode.CutoffRange,
        unwrap_particles=True,
        sort_by_size=True,compute_gyration=True,compute_com=True))

    allsizes = np.empty(0,dtype=int)
    allrgs = np.empty(0)
    allkappas = np.empty(0)
    max_rgs = []
    max_sizes = []
    max_kappas = []
    max_indices = {}
    indices_size = dict([(i,{}) for i in range(1,340)])
    max_cms = np.zeros((traj_mod.source.num_frames,3),dtype=np.float16)
    size_chain = np.zeros((traj_mod.source.num_frames,num_particles),dtype=int)
    coord_num = np.zeros((traj_mod.source.num_frames,400),dtype=int)
    for frame in range(traj_mod.source.num_frames):
        data = pipeline.compute(frame)
        cluster_table = data.tables['clusters']
        cluster_id = cluster_table['Cluster Identifier']
        cluster_size = cluster_table['Cluster Size']
        cluster_tensor = cluster_table['Gyration Tensor']
        cluster_rg = cluster_table['Radius of Gyration']
        cluster_cm = cluster_table['Center of Mass']
        allsizes = np.append(allsizes, np.array(cluster_size).flatten())
        max_sizes.append(cluster_size[0])
        max_rgs.append(cluster_rg[0]/10)
        max_kappas.append(calc_kappa(cluster_tensor[0]))
        coord_num[frame] = np.asarray(data.particles["Coordination"]).astype(int)
        max_indices[frame] = np.where(cluster_id[data.particles["Cluster"]-1]==cluster_id[0])[0]
        max_cms[frame] = cluster_cm[0]
        for i,size in enumerate(cluster_size):
            indices = np.where(cluster_id[data.particles["Cluster"]-1]==cluster_id[i])[0]
            indices_size[size][frame] = np.where(cluster_id[data.particles["Cluster"]-1]==cluster_id[i])[0]
            if (indices>num_particles).all():
                indices -= num_particles
            size_chain[frame,indices] = size
    cluster.loc['size_counts_'+''.join(particle_types)] = np.bincount(allsizes)
    cluster.loc['mean_size_'+''.join(particle_types)] = np.mean(allsizes)
    cluster.loc['max_sizes_'+''.join(particle_types)] = max_sizes
    cluster.loc['max_rgs_'+''.join(particle_types)] = max_rgs
    cluster.loc['max_cms_'+''.join(particle_types)] = max_cms
    cluster.loc['indices_size_'+''.join(particle_types)] = indices_size
    cluster.loc['max_indices_'+''.join(particle_types)] = max_indices
    cluster.loc['max_kappas_'+''.join(particle_types)] = max_kappas
    cluster.loc['size_chain_'+''.join(particle_types)] = size_chain
    cluster.loc['coord_num_'+''.join(particle_types)] = coord_num
    return cluster

cluster = pd.Series(dtype=object)

L = 188
N = 400
ionic = args.ionic
temp = args.temp

t = md.load('cmtraj/{:s}_400_{:d}_{:d}.xtc'.format(args.dirname,temp,ionic),
            top='cmtraj/{:s}_400_{:d}_{:d}.pdb'.format(args.dirname,temp,ionic))

rg = np.load('cmtraj/{:s}_400_{:d}_{:d}_rg.npy'.format(args.dirname,temp,ionic)).mean()
cluster = cluster_analysis(args.dirname,N,temp,ionic,rg,cluster,400,{'B'})

cluster.to_pickle('cmtraj/clusters_{:s}_{:d}.pkl'.format(args.dirname,ionic))
