from simtk import openmm, unit
from simtk.openmm import app
from simtk.openmm import XmlSerializer
from analyse import *
import time
import os
import sys
from argparse import ArgumentParser
import random

parser = ArgumentParser()
parser.add_argument('--proteins',nargs='+',required=True)
parser.add_argument('--numbers',nargs='+',required=True)
parser.add_argument('--temp',nargs='?',const='',type=int,required=True)
parser.add_argument('--dirname',nargs='?',const='',type=str,required=True)
parser.add_argument('--ionic',nargs='?',const='',type=int,required=True)
parser.add_argument('--cutoff',nargs='?',const='', type=float)
args = parser.parse_args()

def simulate(residues,dirname,proteins,composition,temp,ionic,cutoff):
    path = '{:s}/{:d}/{:d}'.format(dirname,temp,ionic)

    residues = residues.set_index('one')

    composition, yukawa_kappa, lj_eps = genParams(residues,proteins,composition,temp,ionic*1e-3)

    # set parameters
    L = 188.
    margin = 1

    system = openmm.System()

    # set box vectors
    a = unit.Quantity(np.zeros([3]), unit.nanometers)
    a[0] = L * unit.nanometers
    b = unit.Quantity(np.zeros([3]), unit.nanometers)
    b[1] = L * unit.nanometers
    c = unit.Quantity(np.zeros([3]), unit.nanometers)
    c[2] = L * unit.nanometers
    system.setDefaultPeriodicBoxVectors(a, b, c)

    # initial config
    z = np.empty(0)
    z = np.append(z,np.random.rand(1)*(L-margin)-(L-margin)/2)
    for z_i in np.random.rand(4000)*(L-margin)-(L-margin)/2:
        z_j = z_i-L if z_i>0 else z_i+L
        if np.all(np.abs(z_i-z_j)>.7):
            z = np.append(z,z_i)
        if z.size == composition.N.sum():
            break

    xy = np.random.rand(composition.N.sum(),2)*(L-margin)-(L-margin)/2

    print('Number of chains',z.size)

    top = md.Topology()
    N_beads = (composition.fasta.apply(lambda x : len(x))*composition.N).values.sum()
    pos = np.empty((N_beads,3))

    start = 0
    begin = 0
    for k,name in enumerate(composition.index):
        N = composition.loc[name].N
        Naa = len(composition.loc[name].fasta)
        for z_i,(x_i,y_i) in zip(z[start:start+N],xy[start:start+N]):
            chain = top.add_chain()
            pos[begin:begin+Naa,:] = xy_spiral_array(Naa,L/2.) + np.array([x_i,y_i,z_i])
            for resname in composition.loc[name].fasta_termini:
                residue = top.add_residue(resname, chain)
                top.add_atom(resname, element=md.element.carbon, residue=residue)
            for i in range(chain.n_atoms-1):
                top.add_bond(chain.atom(i),chain.atom(i+1))
            begin += Naa
        start += N

    md.Trajectory(np.array(pos).reshape(N_beads,3), top, 0, [L,L,L], [90,90,90]).save_pdb(path+'/top.pdb')

    pdb = app.pdbfile.PDBFile(path+'/top.pdb')

    for name in composition.index:
        fasta = composition.loc[name].fasta
        for _ in range(composition.loc[name].N):
            system.addParticle((residues.loc[fasta[0]].MW+2)*unit.amu)
            for a in fasta[1:-1]:
                system.addParticle(residues.loc[a].MW*unit.amu)
            system.addParticle((residues.loc[fasta[-1]].MW+16)*unit.amu)

    hb = openmm.openmm.HarmonicBondForce()
    energy_expression = 'select(step(r-2^(1/6)*s),4*eps*l*((s/r)^12-(s/r)^6-shift),4*eps*((s/r)^12-(s/r)^6-l*shift)+eps*(1-l))'
    ah = openmm.openmm.CustomNonbondedForce(energy_expression+'; s=0.5*(s1+s2); l=0.5*(l1+l2); shift=(0.5*(s1+s2)/rc)^12-(0.5*(s1+s2)/rc)^6')

    ah.addGlobalParameter('eps',lj_eps*unit.kilojoules_per_mole)
    ah.addGlobalParameter('rc',cutoff*unit.nanometer)
    ah.addPerParticleParameter('s')
    ah.addPerParticleParameter('l')

    print('rc',cutoff*unit.nanometer)

    yu = openmm.openmm.CustomNonbondedForce('q*(exp(-kappa*r)/r-shift); q=q1*q2')
    yu.addGlobalParameter('kappa',yukawa_kappa/unit.nanometer)
    yu.addGlobalParameter('shift',np.exp(-yukawa_kappa*4.0)/4.0/unit.nanometer)
    yu.addPerParticleParameter('q')

    begin = 0
    for name in composition.index:
        fasta = composition.loc[name].fasta
        Naa = len(fasta)
        for _ in range(composition.loc[name].N):
            for a,e in zip(fasta,composition.loc[name].charge):
                yu.addParticle([e*unit.nanometer*unit.kilojoules_per_mole])
                ah.addParticle([residues.loc[a].sigmas*unit.nanometer, residues.loc[a].lambdas*unit.dimensionless])
            for i in range(begin,begin+Naa-1):
                hb.addBond(i, i+1, 0.38*unit.nanometer, 8033*unit.kilojoules_per_mole/(unit.nanometer**2))
                yu.addExclusion(i, i+1)
                ah.addExclusion(i, i+1)
            begin += Naa

    print(begin,N_beads)

    yu.setForceGroup(0)
    ah.setForceGroup(1)
    yu.setNonbondedMethod(openmm.openmm.CustomNonbondedForce.CutoffPeriodic)
    ah.setNonbondedMethod(openmm.openmm.CustomNonbondedForce.CutoffPeriodic)
    hb.setUsesPeriodicBoundaryConditions(True)
    yu.setCutoffDistance(4*unit.nanometer)
    ah.setCutoffDistance(cutoff*unit.nanometer)

    system.addForce(hb)
    system.addForce(yu)
    system.addForce(ah)

    integrator = openmm.openmm.LangevinIntegrator(temp*unit.kelvin,0.01/unit.picosecond,0.01*unit.picosecond)

    platform = openmm.Platform.getPlatformByName('CUDA')

    simulation = app.simulation.Simulation(pdb.topology, system, integrator, platform, dict(CudaPrecision='mixed'))

    check_point = path+'/restart.chk'.format(temp)

    if os.path.isfile(check_point):
        print('Reading check point file')
        simulation.loadCheckpoint(check_point)
        simulation.reporters.append(app.dcdreporter.DCDReporter(path+'/t.dcd'.format(name),int(5e5),enforcePeriodicBox=False,append=True))
    else:
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy()
        simulation.reporters.append(app.dcdreporter.DCDReporter(path+'/t.dcd'.format(name),int(5e5),enforcePeriodicBox=False))

    simulation.reporters.append(app.statedatareporter.StateDataReporter(path+'/log',1000000,
             step=True,speed=True,elapsedTime=True,separator='\t'))

    simulation.runForClockTime(60*unit.hour, checkpointFile=check_point, checkpointInterval=2*unit.hour)

    simulation.saveCheckpoint(check_point)

residues = pd.read_csv('residues.csv').set_index('three',drop=False)
proteins = pd.read_csv('proteins.csv',index_col=0)
proteins.fasta = proteins.fasta.apply(list)

composition = pd.DataFrame(index=args.proteins,columns=['N','fasta'])
composition.N = [int(N) for N in args.numbers]
composition.fasta = [proteins.loc[name].fasta for name in composition.index]

t0 = time.time()
simulate(residues,args.dirname,proteins,composition,args.temp,args.ionic,args.cutoff)
print('Timing {:.3f}'.format(time.time()-t0))
