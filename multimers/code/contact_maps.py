import numpy as np
import mdtraj as md
import pandas as pd
from itertools import combinations, product
from mdtraj import element
from argparse import ArgumentParser
import itertools
import os
import time
import string
from scipy.optimize import least_squares

parser = ArgumentParser()
parser.add_argument('--dirname',nargs='?',const='',type=str,required=True)
parser.add_argument('--ionic',nargs='?',const='',type=int,required=True)
parser.add_argument('--temp',nargs='?',const='',type=int,required=True)
parser.add_argument('--size',nargs='?',const='',type=int,required=True)
args = parser.parse_args()

HALR = lambda r,s,l : 4*0.8368*l*((s/r)**12-(s/r)**6)
HASR = lambda r,s,l : 4*0.8368*((s/r)**12-(s/r)**6)+0.8368*(1-l)
HA = lambda r,s,l : np.where(r<2**(1/6)*s, HASR(r,s,l), HALR(r,s,l))
HASP = lambda r,s,l,rc : np.where(r<rc, HA(r,s,l)-HA(rc,s,l), 0)

DH = lambda r,yukawa_eps,lD : yukawa_eps*np.exp(-r/lD)/r
DHSP = lambda r,yukawa_eps,lD,rc : np.where(r<rc, DH(r,yukawa_eps,lD)-DH(rc,yukawa_eps,lD), 0)

def calc_energies(t, chain_index_1, chain_index_2, sigmas, lambdas, yukawa_eps, lD, Naa):
    sel1 = t.top.select('chainid {:d}'.format(chain_index_1))
    sel2 = t.top.select('chainid {:d}'.format(chain_index_2))
    pairs_indices = t.top.select_pairs(sel1,sel2)
    d = md.compute_distances(t,pairs_indices).reshape(Naa,Naa)
    ah_ene = HASP(d,sigmas,lambdas,2.0)
    dh_ene = DHSP(d,yukawa_eps,lD,4.0)
    switch_2 = (.5-.5*np.tanh((d-sigmas)/.2))
    switch_3 = (.5-.5*np.tanh((d-sigmas)/.3))
    return ah_ene, dh_ene, switch_2, switch_3

def max_clust_dist(df_residues,dirname,df_proteins,temp,ionic,size):
    path = '{:s}/{:d}/{:d}'.format(dirname,temp,ionic)

    s_aa = md.load(path+'/top.pdb')

    Lx = s_aa.unitcell_lengths[0,0]
    Ly = s_aa.unitcell_lengths[0,1]
    Lz = s_aa.unitcell_lengths[0,2]

    prot = df_proteins.loc[dirname]
    Naa = len(prot.fasta)

    masses = df_residues.loc[prot.fasta,'MW'].values
    masses[0] += 2
    masses[-1] += 16

    cluster = pd.read_pickle('cmtraj/clusters_{:s}_{:d}.pkl'.format(dirname,ionic))

    edges = np.arange(0,Lx/2,.1)

    x = edges[:-1]+(edges[1]-edges[0])/2.

    pdb = md.load_pdb('cmtraj/{:s}_400_{:d}_{:d}.pdb'.format(dirname,temp,ionic))

    fasta = prot.fasta
    df_residues.loc['H','q'] = 1. / ( 1 + 10**(prot.pH-6) )
    df_residues.loc['X'] = df_residues.loc[fasta[0]]
    df_residues.loc['Z'] = df_residues.loc[fasta[-1]]
    df_residues.loc['X','q'] = df_residues.loc[prot.fasta[0],'q'] + 1.
    df_residues.loc['Z','q'] = df_residues.loc[prot.fasta[-1],'q'] - 1.
    fasta[0] = 'X'
    fasta[-1] = 'Z'

    pairs = np.asarray(list(itertools.product(fasta,fasta)))
    sigmas = 0.5*(df_residues.loc[pairs[:,0]].sigmas.values+df_residues.loc[pairs[:,1]].sigmas.values).reshape(Naa,Naa)
    lambdas = 0.5*(df_residues.loc[pairs[:,0]].lambdas.values+df_residues.loc[pairs[:,1]].lambdas.values).reshape(Naa,Naa)

    RT = 8.3145*temp*1e-3
    fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(temp)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/RT
    qq = (df_residues.loc[pairs[:,0]].q.values*df_residues.loc[pairs[:,1]].q.values).reshape(Naa,Naa)
    yukawa_eps = qq*lB*RT
    lD = 1. / np.sqrt(8*np.pi*lB*ionic/1000.*6.022/10)

    cutoff = np.load('cmtraj/{:s}_400_{:d}_{:d}_rg.npy'.format(dirname,temp,ionic)).mean()*1.5

    ah_ene_mat = np.zeros((Naa,Naa))
    dh_ene_mat = np.zeros((Naa,Naa))
    s_2_mat = np.zeros((Naa,Naa))
    s_3_mat = np.zeros((Naa,Naa))

    rgs = []
    kappas = []
    number_of_chains = []
    counter_mat = 0

    for s in range(size,size+10):
        if s in cluster['indices_size_B']:
            indices_dict = cluster['indices_size_B'][s]
            frames = np.asarray(list(indices_dict.keys()))

            for k,frame in enumerate(frames):
                sel = pdb.top.select(''.join(['index {:d}'.format(i) for i in indices_dict[frame]]))
                t = md.load_frame('cmtraj/{:s}_400_{:d}_{:d}.xtc'.format(dirname,temp,ionic),
                          top='cmtraj/{:s}_400_{:d}_{:d}.pdb'.format(dirname,temp,ionic),
                          atom_indices=sel,index=frame)
                number_of_chains.append( len(sel) )
                vec = np.copy(t.xyz)
                pairs = t.top.select_pairs('all','all')
                d = md.compute_distances(t,pairs,periodic=True)
                bonds = pairs[(d<=cutoff).flatten()]
                for i,j in bonds:
                    t.top.add_bond(t.top.atom(i),t.top.atom(j))
                unique, counts = np.unique(bonds,return_counts=True)

                t_w = t.image_molecules(inplace=False, anchor_molecules=[set(t.top.residue(i).atoms) for i in unique], make_whole=True)

                dist_cm_ndx = np.linalg.norm(t_w.xyz[0]-md.compute_center_of_geometry(t_w),axis=1).argsort()

                vec = vec - t_w.xyz

                sel_aa = s_aa.top.select(''.join(['chainid {:d} '.format(i) for i in indices_dict[frame]]))
                t = md.load_frame(path+'/t.dcd', frame, top=s_aa.top, atom_indices=sel_aa)

                xyz = np.copy(t.xyz)
                N = t.n_atoms//t.n_chains
                for i in range(t.n_chains):
                    xyz[:,i*N:(i+1)*N,:] -= vec[:,i,:]
                t = md.Trajectory(xyz, t.top, time=np.arange(0,1,1),
                        unitcell_lengths=[[Lx,Lx,Lx]],
                        unitcell_angles=[[90,90,90]])

                com = md.compute_center_of_mass(t)
                chain_com = [np.linalg.norm(com-md.compute_center_of_mass(t.atom_slice(t.top.select(f'chainid {i:d}')))) for i in range(t.n_chains)]
                index_1 = np.argmin(chain_com)

                if counter_mat == 0:
                    t.save(f'ene_mat/snapshot_{dirname:s}_{ionic:d}_{s:d}_{index_1:d}.pdb')

                rgs.append(md.compute_rg(t)[0])
                kappas.append(md.relative_shape_antisotropy(t)[0])

                counter_mat += 1.
                for index_2 in np.setdiff1d(np.arange(t.n_chains),[index_1]):
                    ah_ene, dh_ene, s_2, s_3 = calc_energies(t[0],index_1,index_2,sigmas,lambdas,yukawa_eps,lD,Naa)
                    ah_ene_mat += ah_ene
                    dh_ene_mat += dh_ene
                    s_2_mat += s_2
                    s_3_mat += s_3
    if counter_mat > 0:
        np.savetxt('ene_mat/{:s}_size_rg_kappa_{:d}_{:d}_{:d}.dat'.format(dirname,temp,ionic,size),np.c_[number_of_chains,rgs,kappas])
        np.save('ene_mat/{:s}_ah_{:d}_{:d}_{:d}.npy'.format(dirname,temp,ionic,size),ah_ene_mat/counter_mat)
        np.save('ene_mat/{:s}_dh_{:d}_{:d}_{:d}.npy'.format(dirname,temp,ionic,size),dh_ene_mat/counter_mat)
        np.save('ene_mat/{:s}_s2_{:d}_{:d}_{:d}.npy'.format(dirname,temp,ionic,size),s_2_mat/counter_mat)
        np.save('ene_mat/{:s}_s3_{:d}_{:d}_{:d}.npy'.format(dirname,temp,ionic,size),s_3_mat/counter_mat)

residues = pd.read_csv('residues.csv').set_index('one',drop=False)
proteins = pd.read_csv('proteins.csv',index_col=0)
proteins.fasta = proteins.fasta.apply(list)

max_clust_dist(residues,args.dirname,proteins,args.temp,args.ionic,args.size)
