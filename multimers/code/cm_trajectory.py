import numpy as np
import mdtraj as md
import pandas as pd
from itertools import combinations, product
from mdtraj import element
from argparse import ArgumentParser
import logging
import itertools
import os
import time
import string
import gc

parser = ArgumentParser()
parser.add_argument('--proteins',nargs='+',required=True)
parser.add_argument('--numbers',nargs='+',required=True)
parser.add_argument('--temp',nargs='?',const='',type=int,required=True)
parser.add_argument('--dirname',nargs='?',const='',type=str,required=True)
parser.add_argument('--ionic',nargs='?',const='',type=int,required=True)
args = parser.parse_args()

logging.basicConfig(filename='cmtraj/{:s}.log'.format(args.dirname),level=logging.INFO)

def calc_cm_rg(t,masses):
    chain_cm = (np.sum(t.xyz*masses[np.newaxis,:,np.newaxis],axis=1)/masses.sum()).astype(np.float16)
    si = np.linalg.norm(t.xyz - chain_cm[:,np.newaxis,:],axis=2).astype(np.float16)
    chain_rg = np.sqrt(np.sum(si**2*masses,axis=1)/masses.sum()).astype(np.float16)
    return chain_cm, chain_rg

def traj_cm_rg(df_residues,dirname,df_proteins,composition,temp,ionic):
    path = '{:s}/{:d}/{:d}'.format(dirname,temp,ionic)
    t = md.load_pdb(path+'/top.pdb')

    Lx = t.unitcell_lengths[0,0]
    Ly = t.unitcell_lengths[0,1]
    Lz = t.unitcell_lengths[0,2]

    logging.info('Generating Topology')

    top = md.Topology()
    all_masses = []
    for name in composition.index:
        prot = df_proteins.loc[name]
        masses = df_residues.loc[prot.fasta,'MW'].values
        masses[0] += 2
        masses[-1] += 16
        radii = df_residues.loc[prot.fasta,'sigmas'].values/2
        Naa = len(prot.fasta)
        n_chains = composition.loc[name].N
        for _ in range(n_chains):
            all_masses.append(masses)
            chain = top.add_chain()
            residue = top.add_residue('C{:d}'.format(chain.index), chain, resSeq=chain.index)
            for i,resname in enumerate(prot.fasta):
                element.Element._elements_by_symbol.pop('A'+resname, None)
                el = element.Element.__new__(element.Element, 1,
                                         'A'+resname, 'A'+resname, masses[i], radii[i])
                atom = top.add_atom('A'+resname, element=el, residue=residue)
            for i in range(chain.n_atoms-1):
                top.add_bond(chain.atom(i),chain.atom(i+1))

    time0 = time.time()

    cm = []
    rg = []

    t = md.load_dcd(path+'/t.dcd',top=top)

    logging.info('First chunk {:d} frames, {:g} seconds'.format(t.n_frames,time.time()-time0))

    for i in range(composition.N.values.sum()):
        logging.info('chainid {:d}, {:g}'.format(i,time.time()-time0))
        chain_cm, chain_rg = calc_cm_rg(t.atom_slice(t.top.select('chainid {:d}'.format(i))),all_masses[i])
        cm.append(chain_cm)
        rg.append(chain_rg)

    logging.info('Calculation complete, {:g} seconds'.format(time.time()-time0))
    rg = np.asarray(rg)

    gc.collect()

    cm = np.swapaxes(np.asarray(cm),0,1)
    logging.info(cm.shape)

    top = md.Topology()
    chain = top.add_chain()

    counter = 0
    for letter,name,ele in zip(string.ascii_uppercase[1:],composition.index,[md.element.boron,md.element.carbon]):
        for _ in range(composition.loc[name].N):
            res = top.add_residue(letter, chain, resSeq=counter)
            top.add_atom(res.name, element=ele, residue=res)
            counter += 1

    cmtraj = md.Trajectory(cm, top, time=np.arange(0,cm.shape[0],1),
            unitcell_lengths=np.array([[Lx,Ly,Lz]]*cm.shape[0]),
            unitcell_angles=t.unitcell_angles)

    gc.collect()

    del cm

    # save radii of gyration of the chains
    np.save('cmtraj/{:s}_{:d}_{:d}_{:d}_rg.npy'.format(dirname,cmtraj.n_atoms,temp,ionic),rg)
    # save trajectory of the centers of mass of the chains
    cmtraj.save_xtc('cmtraj/{:s}_{:d}_{:d}_{:d}.xtc'.format(dirname,cmtraj.n_atoms,temp,ionic))
    cmtraj[-1].save_gro('cmtraj/{:s}_{:d}_{:d}_{:d}.gro'.format(dirname,cmtraj.n_atoms,temp,ionic))
    cmtraj[-1].save_pdb('cmtraj/{:s}_{:d}_{:d}_{:d}.pdb'.format(dirname,cmtraj.n_atoms,temp,ionic))

residues = pd.read_csv('residues.csv').set_index('one',drop=False)
proteins = pd.read_csv('proteins.csv',index_col=0)
proteins.fasta = proteins.fasta.apply(list)

composition = pd.DataFrame(index=args.proteins,columns=['N','fasta'])
composition.N = [int(N) for N in args.numbers]
composition.fasta = [proteins.loc[name].fasta for name in composition.index]

traj_cm_rg(residues,args.dirname,proteins,composition,args.temp,args.ionic)
