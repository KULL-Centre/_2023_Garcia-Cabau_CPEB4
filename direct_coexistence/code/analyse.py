import pandas as pd
import numpy as np
import mdtraj as md
import itertools
import os
import MDAnalysis
from MDAnalysis import transformations

def initProteins():
    proteins = pd.DataFrame(columns=['pH','ionic','fasta'])
    fasta_CPEB4 = """MGDYGFGVLVQSNTGNKSAFPVRFHPHLQPPHHHQNATPSPAAFINNNTAANGSSAGSAWLFP
APATHNIQDEILGSEKAKSQQQEQQDPLEKQQLSPSPGQEAGILPETEKAKSEENQGDNSSENGNGKEKIRIESPVLTG
FDYQEATGLGTSTQPLTSSASSLTGFSNWSAAIAPSSSTIINEDASFFHQGGVPAASANNGALLFQNFPHHVSPGFGGS
FSPQIGPLSQHHPHHPHFQHHHSQHQQQRRSPASPHPPPFTHRNAAFNQLPHLANNLNKPPSPWSSYQSPSPTPSSSWS
PGGGGYGGWGGSQGRDHRRGLNGGITPLNSISPLKKNFASNHIQLQKYARPSSAFAPKSWMEDSLNRADNIFPFPDRPR
TFDMHSLESSLIDIMRAENDTIKARTYGRRRGQSSLFPMEDGFLDDGRGDQPLHSGLGSPHCFSHQNGE""".replace('\n', '')
    fasta_CPEB4H50S = """MGDYGFGVLVQSNTGNKSAFPVRFSPHLQPPSHSQNATPSPAAFINNNTAANGSSAGSAWLFPAP
ATHNIQDEILGSEKAKSQQQEQQDPLEKQQLSPSPGQEAGILPETEKAKSEENQGDNSSENGNGKEKI
RIESPVLTGFDYQEATGLGTSTQPLTSSASSLTGFSNWSAAIAPSSSTIINEDASFFSQGGVPAASAN
NGALLFQNFPHSVSPGFGGSFSPQIGPLSQHSPHSPHFQSHSSQHQQQRRSPASPSPPPFTHRNAAFN
QLPSLANNLNKPPSPWSSYQSPSPTPSSSWSPGGGGYGGWGGSQGRDHRRGLNGGITPLNSISPLKKN
FASNSIQLQKYARPSSAFAPKSWMEDSLNRADNIFPFPDRPRTFDMHSLESSLIDIMRAENDTIKART
YGRRRGQSSLFPMEDGFLDDGRGDQPLSSGLGSPHCFSSQNGE""".replace('\n','')
    fasta_CPEB4H50S_Clust = """MGDYGFGVLVQSNTGNKSAFPVRFHPHLQPPHHHQNATPSPAAFINNNTAANGSSAGSAWLFPAP
ATHNIQDEILGSEKAKSQQQEQQDPLEKQQLSPSPGQEAGILPETEKAKSEENQGDNSSENGNGKEKI
RIESPVLTGFDYQEATGLGTSTQPLTSSASSLTGFSNWSAAIAPSSSTIINEDASFFHQGGVPAASAN
NGALLFQNFPSSVSPGFGGSFSPQIGPLSQSSPSSPSFQSSSSQSQQQRRSPASPSPPPFTSRNAAFN
QLPSLANNLNKPPSPWSSYQSPSPTPSSSWSPGGGGYGGWGGSQGRDHRRGLNGGITPLNSISPLKKN
FASNHIQLQKYARPSSAFAPKSWMEDSLNRADNIFPFPDRPRTFDMHSLESSLIDIMRAENDTIKART
YGRRRGQSSLFPMEDGFLDDGRGDQPLHSGLGSPHCFSHQNGE""".replace('\n','')
    fasta_CPEB4H25S = """MGDYGFGVLVQSNTGNKSAFPVRFSPHLQPPHHSQNATPSPAAFINNNTAANGSSAGSAWLFPAP
ATHNIQDEILGSEKAKSQQQEQQDPLEKQQLSPSPGQEAGILPETEKAKSEENQGDNSSENGNGKEKI
RIESPVLTGFDYQEATGLGTSTQPLTSSASSLTGFSNWSAAIAPSSSTIINEDASFFHQGGVPAASAN
NGALLFQNFPHSVSPGFGGSFSPQIGPLSQHHPHSPHFQHHSSQHQQQRRSPASPHPPPFTHRNAAFN
QLPSLANNLNKPPSPWSSYQSPSPTPSSSWSPGGGGYGGWGGSQGRDHRRGLNGGITPLNSISPLKKN
FASNHIQLQKYARPSSAFAPKSWMEDSLNRADNIFPFPDRPRTFDMHSLESSLIDIMRAENDTIKART
YGRRRGQSSLFPMEDGFLDDGRGDQPLSSGLGSPHCFSHQNGE""".replace('\n','')
    fasta_CPEB4H25S_Clust = """MGDYGFGVLVQSNTGNKSAFPVRFHPHLQPPHHHQNATPSPAAFINNNTAANGSSAGSAWLFPAP
ATHNIQDEILGSEKAKSQQQEQQDPLEKQQLSPSPGQEAGILPETEKAKSEENQGDNSSENGNGKEKI
RIESPVLTGFDYQEATGLGTSTQPLTSSASSLTGFSNWSAAIAPSSSTIINEDASFFHQGGVPAASAN
NGALLFQNFPSHVSPGFGGSFSPQIGPLSQSHPSHPSFQHSHSQSQQQRRSPASPHPPPFTSRNAAFN
QLPHLANNLNKPPSPWSSYQSPSPTPSSSWSPGGGGYGGWGGSQGRDHRRGLNGGITPLNSISPLKKN
FASNHIQLQKYARPSSAFAPKSWMEDSLNRADNIFPFPDRPRTFDMHSLESSLIDIMRAENDTIKART
YGRRRGQSSLFPMEDGFLDDGRGDQPLHSGLGSPHCFSHQNGE""".replace('\n','')

    proteins.loc['CPEB4'] = dict(pH=8.0,fasta=fasta_CPEB4,ionic=0.15)
    proteins.loc['CPEB4H50S'] = dict(pH=8.0,fasta=fasta_CPEB4H50S,ionic=0.15)
    proteins.loc['CPEB4H50S_Clust'] = dict(pH=8.0,fasta=fasta_CPEB4H50S_Clust,ionic=0.15)
    proteins.loc['CPEB4H25S'] = dict(pH=8.0,fasta=fasta_CPEB4H25S,ionic=0.15)
    proteins.loc['CPEB4H25S_Clust'] = dict(pH=8.0,fasta=fasta_CPEB4H25S_Clust,ionic=0.15)
    proteins.loc['CPEB4pH7'] = dict(pH=7.0,fasta=fasta_CPEB4,ionic=0.15)
    proteins.loc['CPEB4pH6'] = dict(pH=6.0,fasta=fasta_CPEB4,ionic=0.15)
    return proteins

def genParamsLJ(df,name,prot):
    fasta = prot.fasta.copy()
    r = df.copy()
    r.loc['X'] = r.loc[fasta[0]]
    r.loc['Z'] = r.loc[fasta[-1]]
    r.loc['X','MW'] += 2
    r.loc['Z','MW'] += 16
    fasta[0] = 'X'
    fasta[-1] = 'Z'
    types = list(np.unique(fasta))
    MWs = [r.loc[a,'MW'] for a in types]
    lj_eps = 0.2*4.184
    return lj_eps, fasta, types, MWs

def genParamsDH(df,name,prot,temp):
    kT = 8.3145*temp*1e-3
    fasta = prot.fasta.copy()
    r = df.copy()
    # Set the charge on HIS based on the pH of the protein solution
    r.loc['H','q'] = 1. / ( 1 + 10**(prot.pH-6) )
    r.loc['X'] = r.loc[fasta[0]]
    r.loc['Z'] = r.loc[fasta[-1]]
    fasta[0] = 'X'
    fasta[-1] = 'Z'
    r.loc['X','q'] = r.loc[prot.fasta[0],'q'] + 1.
    r.loc['Z','q'] = r.loc[prot.fasta[-1],'q'] - 1.
    # Calculate the prefactor for the Yukawa potential
    fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(temp)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/kT
    yukawa_eps = [r.loc[a].q*np.sqrt(lB*kT) for a in fasta]
    # Calculate the inverse of the Debye length
    yukawa_kappa = np.sqrt(8*np.pi*lB*prot.ionic*6.022/10)
    return yukawa_eps, yukawa_kappa, r

def calc_zpatch(z,h):
    cutoff = 0
    ct = 0.
    ct_max = 0.
    zwindow = []
    hwindow = []
    zpatch = []
    hpatch = []
    for ix, x in enumerate(h):
        if x > cutoff:
            ct += x
            zwindow.append(z[ix])
            hwindow.append(x)
        else:
            if ct > ct_max:
                ct_max = ct
                zpatch = zwindow
                hpatch = hwindow
            ct = 0.
            zwindow = []
            hwindow = []
    zpatch = np.array(zpatch)
    hpatch = np.array(hpatch)
    return zpatch, hpatch

def center_slab(name,replica,start=None,end=None,step=1,input_pdb='top.pdb'):
    path = f'{name:s}/{replica:d}'
    u = MDAnalysis.Universe(f'{path:s}/{input_pdb:s}',f'{path:s}/{name:s}.dcd',in_memory=True)
    n_frames = len(u.trajectory[start:end:step])
    ag = u.atoms
    n_atoms = ag.n_atoms
    print(name,replica)
    L = u.dimensions[0]/10
    lz = u.dimensions[2]
    edges = np.arange(0,lz+1,1)
    dz = (edges[1] - edges[0]) / 2.
    z = edges[:-1] + dz
    n_bins = len(z)
    hs = np.zeros((n_frames,n_bins))
    with MDAnalysis.Writer(f'{path:s}/traj.dcd',n_atoms) as W:
        for t,ts in enumerate(u.trajectory[start:end:step]):
            # shift max density to center
            zpos = ag.positions.T[2]
            h, e = np.histogram(zpos,bins=edges)
            zmax = z[np.argmax(h)]
            ag.translate(np.array([0,0,-zmax+0.5*lz]))
            ts = transformations.wrap(ag)(ts)
            zpos = ag.positions.T[2]
            h, e = np.histogram(zpos, bins=edges)
            zpatch, hpatch = calc_zpatch(z,h)
            zmid = np.average(zpatch,weights=hpatch)
            ag.translate(np.array([0,0,-zmid+0.5*lz]))
            ts = transformations.wrap(ag)(ts)
            zpos = ag.positions.T[2]
            h, e = np.histogram(zpos,bins=edges)
            hs[t] = h
            W.write(ag)
    np.save(f'{name:s}_{replica:d}.npy',hs,allow_pickle=False)
    return hs, z

def genDCD(name,replica,start=None,end=None,step=1,input_pdb='top.pdb'):
    path = f'{name:s}/{replica:d}'
    t = md.load(f'{path:s}/{name:s}.dcd',top=f'{path:s}/{input_pdb:s}')
    t.xyz *= 10
    t.unitcell_lengths *= 10
    lz = t.unitcell_lengths[0,2]
    edges = np.arange(-lz/2.,lz/2.,1)
    dz = (edges[1]-edges[0])/2.
    z = edges[:-1]+dz
    h = np.apply_along_axis(lambda a: np.histogram(a,bins=edges)[0], 1, t.xyz[:,:,2])
    zmid = np.apply_along_axis(lambda a: z[a.argmax()], 1, h)
    indices = np.argmin(np.abs(t.xyz[:,:,2]-zmid[:,np.newaxis]),axis=1)
    t.save_dcd(path+'/traj4.dcd')

    u = MDAnalysis.Universe(path+'/top.pdb',path+'/traj4.dcd')
    ag = u.atoms
    with MDAnalysis.Writer(path+'/traj3.dcd', ag.n_atoms) as W:
        for ts,ndx in zip(u.trajectory,indices):
            ts = transformations.unwrap(ag)(ts)
            ts = transformations.center_in_box(
                u.select_atoms('index {:d}'.format(ndx)), center='geometry')(ts)
            ts = transformations.wrap(ag)(ts)
            W.write(ag)

    t = md.load(path+'/traj3.dcd',top=path+'/top.pdb')
    edges = np.arange(0,lz,1)
    dz = (edges[1]-edges[0])/2.
    z = edges[:-1]+dz
    h = np.apply_along_axis(lambda a: np.histogram(a,bins=edges)[0], 1, t.xyz[:,:,2])
    h = np.mean(h[:120],axis=0)
    maxoverlap = np.apply_along_axis(lambda a: np.correlate(h,np.histogram(a,
                bins=edges)[0], 'full').argmax()-h.size+dz, 1, t.xyz[:,:,2])

    u = MDAnalysis.Universe(path+'/top.pdb',path+'/traj3.dcd')
    ag = u.atoms
    with MDAnalysis.Writer(path+'/traj2.dcd', ag.n_atoms) as W:
        for ts,mo in zip(u.trajectory,maxoverlap):
            ts = transformations.unwrap(ag)(ts)
            ts = transformations.translate([0,0,mo*10])(ts)
            ts = transformations.wrap(ag)(ts)
            W.write(ag)

    t = md.load(path+'/traj2.dcd',top=path+'/top.pdb')
    h = np.apply_along_axis(lambda a: np.histogram(a,bins=edges)[0], 1, t.xyz[:,:,2])
    zmid = np.apply_along_axis(lambda a: z[a>np.quantile(a,.98)].mean(), 1, h)
    indices = np.argmin(np.abs(t.xyz[:,:,2]-zmid[:,np.newaxis]),axis=1)

    u = MDAnalysis.Universe(path+'/top.pdb',path+'/traj2.dcd')
    ag = u.atoms
    with MDAnalysis.Writer(path+'/traj1.dcd', ag.n_atoms) as W:
        for ts,ndx in zip(u.trajectory,indices):
            ts = transformations.unwrap(ag)(ts)
            ts = transformations.center_in_box(
                u.select_atoms('index {:d}'.format(ndx)), center='geometry')(ts)
            ts = transformations.wrap(ag)(ts)
            W.write(ag)

    t = md.load(path+'/traj1.dcd',top=path+'/top.pdb')
    h = np.apply_along_axis(lambda a: np.histogram(a,bins=edges)[0], 1, t.xyz[:,:,2])
    h = np.mean(h[120:],axis=0)
    maxoverlap = np.apply_along_axis(lambda a: np.correlate(h,np.histogram(a,
                bins=edges)[0], 'full').argmax()-h.size+dz, 1, t.xyz[:,:,2])

    u = MDAnalysis.Universe(path+'/top.pdb',path+'/traj1.dcd')
    ag = u.atoms
    with MDAnalysis.Writer(path+'/traj0.dcd', ag.n_atoms) as W:
        for ts,mo in zip(u.trajectory,maxoverlap):
            ts = transformations.unwrap(ag)(ts)
            ts = transformations.translate([0,0,mo*10])(ts)
            ts = transformations.wrap(ag)(ts)
            W.write(ag)

    t = md.load(path+'/traj0.dcd',top=path+'/top.pdb')

    h = np.apply_along_axis(lambda a: np.histogram(a,bins=edges)[0], 1, t.xyz[:,:,2])
    np.save(f'{name:s}_{replica:d}.npy',h,allow_pickle=False)
    os.remove(path+'/traj1.dcd')
    os.remove(path+'/traj2.dcd')
    os.remove(path+'/traj3.dcd')
    os.remove(path+'/traj4.dcd')
    t.xyz /= 10
    t.unitcell_lengths /= 10
    t[0].save_pdb(path+'/top.pdb')
    t.save_dcd(path+'/traj0.dcd')
