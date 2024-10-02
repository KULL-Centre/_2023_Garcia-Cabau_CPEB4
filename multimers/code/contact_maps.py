import numpy as np
import mdtraj as md
import pandas as pd
from argparse import ArgumentParser
import itertools
import logging

parser = ArgumentParser()
parser.add_argument('--dirname',nargs='?',const='',type=str,required=True)
parser.add_argument('--ionic',nargs='?',const='',type=int,required=True)
parser.add_argument('--temp',nargs='?',const='',type=int,required=True)
parser.add_argument('--size',nargs='?',const='',type=int,required=True)
args = parser.parse_args()

logging.basicConfig(filename=f'{args.size:d}.log',level=logging.INFO)

def contact_map(df_residues,dirname,df_proteins,temp,ionic,size):
    path = '{:s}/{:d}/{:d}'.format(dirname,temp,ionic)

    s_aa = md.load(path+'/top.pdb')

    Lx = s_aa.unitcell_lengths[0,0]

    prot = df_proteins.loc[dirname]
    Naa = len(prot.fasta)

    cluster = pd.read_pickle('cmtraj/clusters_{:s}_{:d}.pkl'.format(dirname,ionic))

    cutoff = np.load('cmtraj/{:s}_400_{:d}_{:d}_rg.npy'.format(dirname,temp,ionic)).mean()*1.5

    cmap = np.zeros((1,Naa*Naa*(size-1)))

    rgs = []
    kappas = []
    number_of_chains = []
    counter_map = 0

    t_cg = md.load_dcd(path+'/t.dcd', top=path+'/top.pdb')
    t_cm = md.load_xtc(f'cmtraj/{dirname:s}_400_{temp:d}_{ionic:d}.xtc',top=f'cmtraj/{dirname:s}_400_{temp:d}_{ionic:d}.pdb')

    if size in cluster['indices_size_B']:
        indices_dict = cluster['indices_size_B'][size]
        frames = np.asarray(list(indices_dict.keys()))

        for k,frame in enumerate(frames):
            logging.info(f'frame {k:d} of {frames.size:d}')
            sel = pdb.top.select(''.join(['index {:d}'.format(i) for i in indices_dict[frame]]))
            t = t_cm[frame].atom_slice(sel)
            #t = md.load_frame('cmtraj/{:s}_400_{:d}_{:d}.xtc'.format(dirname,temp,ionic),
            #          top='cmtraj/{:s}_400_{:d}_{:d}.pdb'.format(dirname,temp,ionic),
            #          atom_indices=sel,index=frame)
            number_of_chains.append( len(sel) )
            vec = np.copy(t.xyz)
            pairs = t.top.select_pairs('all','all')
            d = md.compute_distances(t,pairs,periodic=True)
            bonds = pairs[(d<=cutoff).flatten()]
            for i,j in bonds:
                t.top.add_bond(t.top.atom(i),t.top.atom(j))
            unique, counts = np.unique(bonds,return_counts=True)

            t_w = t.image_molecules(inplace=False, anchor_molecules=[set(t.top.residue(i).atoms) for i in unique], make_whole=True)

            vec = vec - t_w.xyz

            sel_aa = s_aa.top.select(''.join(['chainid {:d} '.format(i) for i in indices_dict[frame]]))
            #t = md.load_frame(path+'/t.dcd', frame, top=s_aa.top, atom_indices=sel_aa)
            t = t_cg[frame].atom_slice(sel_aa)

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

            if counter_map == 0:
                t.save(f'contact_maps/snapshot_{dirname:s}_{ionic:d}_{size:d}_{index_1:d}.pdb')

            rgs.append(md.compute_rg(t)[0])
            kappas.append(md.relative_shape_antisotropy(t)[0])

            pair_indices = t.top.select_pairs(f'chainid {index_1:d}',f'not chainid {index_1:d}')

            counter_map += 1.
            d = md.compute_distances(t,pair_indices)
            cmap += .5-.5*np.tanh((d-1.)/.3)

    cmap = cmap.reshape(448,-1,448).sum(axis=1)

    if counter_map > 0:
        np.savetxt('contact_maps/{:s}_size_rg_kappa_{:d}_{:d}_{:d}.dat'.format(dirname,temp,ionic,size),np.c_[number_of_chains,rgs,kappas])
        np.save('contact_maps/{:s}_contacts_{:d}_{:d}_{:d}.npy'.format(dirname,temp,ionic,size),cmap/counter_map)

residues = pd.read_csv('residues.csv').set_index('one',drop=False)
proteins = pd.read_csv('proteins.csv',index_col=0)
proteins.fasta = proteins.fasta.apply(list)

contact_map(residues,args.dirname,proteins,args.temp,args.ionic,args.size)
