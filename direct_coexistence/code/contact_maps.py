from analyse import *
import MDAnalysis
from MDAnalysis import transformations
import time
import os
import glob
import sys
import pandas as pd
import numpy as np
import mdtraj as md
import itertools
from mdtraj import element
from argparse import ArgumentParser
from scipy.optimize import least_squares
import time

parser = ArgumentParser()
parser.add_argument('--name',nargs='?',const='', type=str)
parser.add_argument('--temp',nargs='?',const='', type=int)
parser.add_argument('--chunk',nargs='?',const='', type=int)
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
    d = md.compute_distances(t,pairs_indices).reshape(t.n_frames,Naa,Naa)
    ah_ene = HASP(d,sigmas,lambdas,2.0)
    dh_ene = DHSP(d,yukawa_eps,lD,4.0)
    switch_2 = (.5-.5*np.tanh((d-sigmas)/.2))
    switch_3 = (.5-.5*np.tanh((d-sigmas)/.3))
    return ah_ene, dh_ene, switch_2, switch_3

def calc_cm_rg(t,masses):
    chain_cm = (np.sum(t.xyz*masses[np.newaxis,:,np.newaxis],axis=1)/masses.sum()).astype(np.float16)
    si = np.linalg.norm(t.xyz - chain_cm[:,np.newaxis,:],axis=2).astype(np.float16)
    chain_rg = np.sqrt(np.sum(si**2*masses,axis=1)/masses.sum()).astype(np.float16)
    chain_ete = md.compute_distances(t, [[0,t.n_atoms-1]]).flatten()
    return chain_cm, chain_rg, chain_ete

def calcWidth(path,name,temp):
    # this function finds the z-positions that delimit the slab and the dilute phase
    h = np.load('{:s}_{:d}.npy'.format(name,temp),allow_pickle=False)
    lz = (h.shape[1]+1)
    edges = np.arange(-lz/2.,lz/2.,1)/10
    dz = (edges[1]-edges[0])/2.
    z = edges[:-1]+dz
    profile = lambda x,a,b,c,d : .5*(a+b)+.5*(b-a)*np.tanh((np.abs(x)-c)/d)
    residuals = lambda params,*args : ( args[1] - profile(args[0], *params) )
    hm = np.mean(h[100:],axis=0)
    z1 = z[z>0]
    h1 = hm[z>0]
    z2 = z[z<0]
    h2 = hm[z<0]
    p0=[hm.min(),hm.max(),3,1]
    res1 = least_squares(residuals, x0=p0, args=[z1, h1], bounds=([0]*4,[1e3]*4))
    res2 = least_squares(residuals, x0=p0, args=[z2, h2], bounds=([0]*4,[1e3]*4))
    cutoff1 = .5*(np.abs(res1.x[2]-.5*res1.x[3])+np.abs(-res2.x[2]+.5*res2.x[3]))
    cutoff2 = .5*(np.abs(res1.x[2]+6*res1.x[3])+np.abs(-res2.x[2]-6*res2.x[3]))
    return cutoff1, cutoff2, z, edges

def analyse_traj(df,proteins,name,temp,chunk):
    # this function finds the index of the chain at the center of the slab for each frame
    path = '{:s}/{:d}/'.format(name,temp)
    cutoff1, cutoff2, z, edges = calcWidth(path,name,temp)
    print(name,cutoff1,cutoff2)
    prot = proteins.loc[name]
    Naa = len(prot.fasta)
    masses = df.loc[prot.fasta,'MW'].values
    masses[0] += 2
    masses[-1] += 16
    radii = df.loc[prot.fasta,'sigmas'].values/2

    t = md.load_pdb(path+'top.pdb')

    n_chains = int( t.n_atoms / Naa )

    top = md.Topology()
    for _ in range(n_chains):
        chain = top.add_chain()
        for resname in prot.fasta:
            residue = top.add_residue(df.loc[resname,'three'], chain)
            top.add_atom(df.loc[resname,'three'],
                         element=md.element.carbon, residue=residue)
        for i in range(chain.n_atoms-1):
            top.add_bond(chain.atom(i),chain.atom(i+1))

    t = md.load_dcd(path+'traj.dcd',top)

    print(t.n_frames)

    t.xyz -= t.unitcell_lengths[0,:]/2

    t.make_molecules_whole(inplace=True)
    t = t[200:] # skip first 1 us
    t = t[t.n_frames%10:]
    begin = int(t.n_frames/10 * chunk)
    end = int(t.n_frames/10 * (chunk + 1) - 1)

    print(begin,end)

    t = t[begin:end]

    print(t.n_frames)

    h_res = np.zeros((Naa+1,edges.size-1))
    cm_z = np.empty(0)
    indices = np.zeros((n_chains,t.n_frames))
    middle_dist = np.zeros((n_chains,t.n_frames))

    for i in range(n_chains):
        t_chain = t.atom_slice(t.top.select('chainid {:d}'.format(i)))
        chain_cm, chain_rg, chain_ete = calc_cm_rg(t_chain,masses)
        mask_in = np.abs(chain_cm[:,2])<cutoff1
        mask_out = np.abs(chain_cm[:,2])>cutoff2
        cm_z = np.append(cm_z, chain_cm[:,2])
        indices[i] = mask_in
        middle_dist[i] = np.abs(chain_cm[:,2])

    middle_chain = np.argmin(middle_dist,axis=0) # indices of chains at the center of the slab
    del middle_dist

    fasta = prot.fasta
    df.loc['H','q'] = 1. / ( 1 + 10**(prot.pH-6) )
    df.loc['X'] = df.loc[fasta[0]]
    df.loc['Z'] = df.loc[fasta[-1]]
    df.loc['X','q'] = df.loc[prot.fasta[0],'q'] + 1.
    df.loc['Z','q'] = df.loc[prot.fasta[-1],'q'] - 1.
    fasta[0] = 'X'
    fasta[-1] = 'Z'

    pairs = np.asarray(list(itertools.product(fasta,fasta)))
    sigmas = 0.5*(df.loc[pairs[:,0]].sigmas.values+df.loc[pairs[:,1]].sigmas.values).reshape(Naa,Naa)
    lambdas = 0.5*(df.loc[pairs[:,0]].lambdas.values+df.loc[pairs[:,1]].lambdas.values).reshape(Naa,Naa)

    RT = 8.3145*temp*1e-3
    fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(temp)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/RT
    qq = (df.loc[pairs[:,0]].q.values*df.loc[pairs[:,1]].q.values).reshape(Naa,Naa)
    yukawa_eps = qq*lB*RT
    lD = 1. / np.sqrt(8*np.pi*lB*prot.ionic*6.022/10)

    top_snap = md.Topology()
    for _ in range(2):
        chain = top_snap.add_chain()
        for i,resname in enumerate(prot.fasta):
            if i>=71 and i<147:
                residue = top_snap.add_residue(df.loc['E','three'], chain)
                top_snap.add_atom(df.loc['E','three'],
                             element=md.element.carbon, residue=residue)
            elif i>=228 and i<252:
                residue = top_snap.add_residue(df.loc['H','three'], chain)
                top_snap.add_atom(df.loc['H','three'],
                             element=md.element.carbon, residue=residue)
            elif i>=402 and i<410:
                residue = top_snap.add_residue(df.loc['R','three'], chain)
                top_snap.add_atom(df.loc['R','three'],
                             element=md.element.carbon, residue=residue)
            else:
                residue = top_snap.add_residue(df.loc['A','three'], chain)
                top_snap.add_atom(df.loc['A','three'],
                             element=md.element.carbon, residue=residue)
        for i in range(chain.n_atoms-1):
            top_snap.add_bond(chain.atom(i),chain.atom(i+1))

    ah_mat = np.zeros((t.n_frames,Naa,Naa))
    dh_mat = np.zeros((t.n_frames,Naa,Naa))
    s2_mat = np.zeros((t.n_frames,Naa,Naa))
    s3_mat = np.zeros((t.n_frames,Naa,Naa))

    xyz = np.empty(0)

    for chain_1 in np.unique(middle_chain):
        print(chain_1)
        for chain_2 in np.setdiff1d(np.arange(n_chains),[chain_1]):
            ndx = ((middle_chain==chain_1)*indices[chain_2]).astype(bool)
            if np.any(ndx):
                ah_ene, dh_ene, s_2, s_3 = calc_energies(t[ndx],chain_1,chain_2,sigmas,lambdas,yukawa_eps,lD,Naa)
                ah_mat[ndx,:] += ah_ene
                dh_mat[ndx,:] += dh_ene
                s2_mat[ndx,:] += s_2
                s3_mat[ndx,:] += s_3

                # collect snapshots showing contacts between 72-147, HClust, and me4
                threshold = s_3.max()*.5
                if threshold > 0.2:
                    if name == 'CPEB4':
                        ndx_contact_1 = np.unique(np.where(s_3[:,228:252,402:410]>threshold)[0])
                        ndx_contact_2 = np.unique(np.where(s_3[:,402:410,228:252]>threshold)[0])
                    if name == 'CPEB4pH6' or name == 'CPEB4pH7':
                        ndx_contact_1 = np.unique(np.where(s_3[:,71:147,228:252]>threshold)[0])
                        ndx_contact_2 = np.unique(np.where(s_3[:,228:252,71:147]>threshold)[0])
                    ndx_contact = np.unique(np.concatenate([ndx_contact_1,ndx_contact_2]))
                    if ndx_contact.size > 0:
                        sel_chains = t.top.select(''.join(['chainid {:d} '.format(i) for i in [chain_1,chain_2]]))
                        xyz = np.append(xyz, t[ndx].atom_slice(sel_chains).slice(ndx_contact).xyz)

    # save snapshots
    xyz = np.array(xyz).reshape(-1,top_snap.n_atoms,3)
    n_frames = xyz.shape[0]
    t_snap = md.Trajectory(xyz, top_snap, time=np.arange(0,n_frames,1),
         unitcell_lengths=[[25,25,300]]*n_frames,
         unitcell_angles=[[90,90,90]]*n_frames)
    t_snap = t_snap.image_molecules(inplace=False, anchor_molecules=[set(t.top.residue(i).atoms) for i in range(1)], make_whole=True)
    t_snap.save(f'snapshots/{name:s}_{temp:d}_{chunk:d}.xtc')
    t_snap[0].save(f'snapshots/{name:s}_{temp:d}_{chunk:d}.pdb')

    # save energy and contact maps
    np.save(f's_mat/{name:s}_{temp:d}_{chunk:d}_ah_mat.npy',ah_mat.mean(axis=0))
    np.save(f's_mat/{name:s}_{temp:d}_{chunk:d}_dh_mat.npy',dh_mat.mean(axis=0))
    np.save(f's_mat/{name:s}_{temp:d}_{chunk:d}_s2_mat.npy',s2_mat.mean(axis=0))
    np.save(f's_mat/{name:s}_{temp:d}_{chunk:d}_s3_mat.npy',s3_mat.mean(axis=0))

df = pd.read_csv('residues.csv').set_index('three',drop=False).set_index('one')
proteins = pd.read_csv('proteins.csv',index_col=0)
proteins.fasta = proteins.fasta.apply(list)

t0 = time.time()

analyse_traj(df,proteins,args.name,args.temp,args.chunk)
print('Timing {:.3f}'.format(time.time()-t0))
