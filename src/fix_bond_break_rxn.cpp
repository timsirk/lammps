/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "math.h"
#include "mpi.h"
#include "string.h"
#include "stdlib.h"
#include "fix_bond_break_rxn.h"
#include "update.h"
#include "respa.h"
#include "atom.h"
#include "atom_vec.h"
#include "force.h"
#include "comm.h"
#include "neighbor.h"
#include "domain.h"
#include "random_mars.h"
#include "memory.h"
#include "error.h"
#include "molecule.h"
#include "topo.h"

using namespace LAMMPS_NS;
using namespace FixConst;

#define DELTA 16

/* ---------------------------------------------------------------------- */

FixBondBreakRxn::FixBondBreakRxn(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 6) error->all(FLERR,"Illegal fix bond/break command");

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  nevery = force->inumeric(FLERR,arg[3]);
  if (nevery <= 0) error->all(FLERR,"Illegal fix bond/break command");

  topoflag = 0;
  force_reneighbor = 1;
  next_reneighbor = -1;
  vector_flag = 1;
  size_vector = 2;
  global_freq = 1;
  extvector = 0;

  btype = force->inumeric(FLERR,arg[4]);
  cutoff = force->numeric(FLERR,arg[5]);

  if (btype < 1 || btype > atom->nbondtypes)
    error->all(FLERR,"Invalid bond type in fix bond/break command");
  if (cutoff < 0.0) error->all(FLERR,"Illegal fix bond/break command");

  cutsq = cutoff*cutoff;

  // optional keywords

  fraction = 1.0;
  int seed = 12345;

  int iarg = 6;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"prob") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix bond/break command");
      fraction = force->numeric(FLERR,arg[iarg+1]);
      seed = force->inumeric(FLERR,arg[iarg+2]);
      if (fraction < 0.0 || fraction > 1.0)
        error->all(FLERR,"Illegal fix bond/break/rxn command");
      if (seed <= 0) error->all(FLERR,"Illegal fix bond/break command");
      iarg += 3;
    } else if (strcmp(arg[iarg],"mol") == 0) {
      int imol = atom->find_molecule(arg[iarg+1]);
      if (imol == -1) error->all(FLERR,"Molecule template ID for "
                                 "bond/break/rxn does not exist");
      topoflag = 1;
      onemol = atom->molecules[imol];
      iarg += 2;
    } else error->all(FLERR,"Illegal fix bond/break command");
  }
  // error check

  if (!topoflag)
    error->all(FLERR,"No rxn template was found");
  if (atom->molecular != 1)
    error->all(FLERR,"Cannot use fix bond/break with non-molecular systems");

  // initialize Marsaglia RNG with processor-unique seed

  random = new RanMars(lmp,seed + me);
  topo = new Topo(lmp);

  // set comm sizes needed by this fix
  // topo class does the special comm 

  comm_forward = 2;
  comm_reverse = 2;

  // allocate arrays local to this fix

  nmax = 0;
  partner = finalpartner = NULL;
  distsq = NULL;

  maxbreak = 0;
  broken = NULL;

  // zero out stats
  breakcount = 0;
  breakcounttotal = 0;
}

/* ---------------------------------------------------------------------- */

FixBondBreakRxn::~FixBondBreakRxn()
{
  delete random;
  delete topo;

  // delete locally stored arrays

  memory->destroy(partner);
  memory->destroy(finalpartner);
  memory->destroy(distsq);
  memory->destroy(broken);
}

/* ---------------------------------------------------------------------- */

int FixBondBreakRxn::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= POST_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixBondBreakRxn::init()
{
  if (strstr(update->integrate_style,"respa"))
    nlevels_respa = ((Respa *) update->integrate)->nlevels;

  if (force->improper) {
    if (force->improper_match("class2") || force->improper_match("ring"))
      error->all(FLERR,"Cannot yet use fix bond/break with this "
                 "improper style");
  }

  lastcheck = -1;
}

/* ---------------------------------------------------------------------- */

void FixBondBreakRxn::post_integrate()
{
  int i,j,k,m,n,i1,i2,n1,n3,type;
  double delx,dely,delz,rsq;
  tagint *slist;

  if (update->ntimestep % nevery) return;

  // check that all procs have needed ghost atoms within ghost cutoff
  // only if neighbor list has changed since last check

  if (lastcheck < neighbor->lastcall) check_ghosts();

  // acquire updated ghost atom positions
  // necessary b/c are calling this after integrate, but before Verlet comm

  comm->forward_comm();

  // resize bond partner list and initialize it
  // probability array overlays distsq array
  // needs to be atom->nmax in length

  if (atom->nmax > nmax) {
    memory->destroy(partner);
    memory->destroy(finalpartner);
    memory->destroy(distsq);
    nmax = atom->nmax;
    memory->create(partner,nmax,"bond/break:partner");
    memory->create(finalpartner,nmax,"bond/break:finalpartner");
    memory->create(distsq,nmax,"bond/break:distsq");
    probability = distsq;
  }

  int nlocal = atom->nlocal;
  int nall = atom->nlocal + atom->nghost;

  for (i = 0; i < nall; i++) {
    partner[i] = 0;
    finalpartner[i] = 0;
    distsq[i] = 0.0;
  }

  // loop over bond list
  // setup possible partner list of bonds to break

  double **x = atom->x;
  tagint *tag = atom->tag;
  int *mask = atom->mask;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;

  for (n = 0; n < nbondlist; n++) {
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    type = bondlist[n][2];
    if (!(mask[i1] & groupbit)) continue;
    if (!(mask[i2] & groupbit)) continue;
    if (type != btype) continue;

    delx = x[i1][0] - x[i2][0];
    dely = x[i1][1] - x[i2][1];
    delz = x[i1][2] - x[i2][2];
    rsq = delx*delx + dely*dely + delz*delz;
    if (rsq <= cutsq) continue;

    if (rsq > distsq[i1]) {
      partner[i1] = tag[i2];
      distsq[i1] = rsq;
    }
    if (rsq > distsq[i2]) {
      partner[i2] = tag[i1];
      distsq[i2] = rsq;
    }
  }

  // reverse comm of partner info

  if (force->newton_bond) comm->reverse_comm_fix(this);

  // each atom now knows its winning partner
  // for prob check, generate random value for each atom with a bond partner
  // forward comm of partner and random value, so ghosts have it

  if (fraction < 1.0) {
    for (i = 0; i < nlocal; i++)
      if (partner[i]) probability[i] = random->uniform();
  }

  commflag = 1;
  comm->forward_comm_fix(this,2);

  // break bonds
  // if both atoms list each other as winning bond partner
  // and probability constraint is satisfied

  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  nbreak = 0;
  for (i = 0; i < nlocal; i++) {
    if (partner[i] == 0) continue;
    j = atom->map(partner[i]);
    if (partner[j] != tag[i]) continue;

    // apply probability constraint using RN for atom with smallest ID

    if (fraction < 1.0) {
      if (tag[i] < tag[j]) {
        if (probability[i] >= fraction) continue;
      } else {
        if (probability[j] >= fraction) continue;
      }
    }

    // store final broken bond partners and count the broken bond once
    finalpartner[i] = tag[j];
    finalpartner[j] = tag[i];
    if (tag[i] < tag[j]) nbreak++;
  }

  // tally stats

  MPI_Allreduce(&nbreak,&breakcount,1,MPI_INT,MPI_SUM,world);
  breakcounttotal += breakcount;

  // trigger reneighboring if any bonds were broken
  // this insures neigh lists will immediately reflect the topology changes
  // done if no bonds broken

  if (breakcount) next_reneighbor = update->ntimestep;
  if (!breakcount) return;

  // communicate final partner and 1-2 special neighbors
  // 1-2 neighs already reflect broken bonds

  commflag = 2;
  comm->forward_comm_fix(this);

  // topo handles the rest
  // actually break the bonds
  // update topology and influenced atoms
    topo->change_bonds(0, finalpartner, onemol);
    return;
}

/* ----------------------------------------------------------------------
   insure all atoms 2 hops away from owned atoms are in ghost list
   this allows dihedral 1-2-3-4 to be properly deleted
     and special list of 1 to be properly updated
   if I own atom 1, but not 2,3,4, and bond 3-4 is deleted
     then 2,3 will be ghosts and 3 will store 4 as its finalpartner
------------------------------------------------------------------------- */

void FixBondBreakRxn::check_ghosts()
{
  int i,j,n;
  tagint *slist;

  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int nlocal = atom->nlocal;

  int flag = 0;
  for (i = 0; i < nlocal; i++) {
    slist = special[i];
    n = nspecial[i][1];
    for (j = 0; j < n; j++)
      if (atom->map(slist[j]) < 0) flag = 1;
  }

  int flagall;
  MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);
  if (flagall) 
    error->all(FLERR,"Fix bond/break needs ghost atoms from further away");
  lastcheck = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

void FixBondBreakRxn::post_integrate_respa(int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) post_integrate();
}

/* ---------------------------------------------------------------------- */

int FixBondBreakRxn::pack_forward_comm(int n, int *list, double *buf,
                                    int pbc_flag, int *pbc)
{
  int i,j,k,m,ns;

  if (commflag == 1) {
    m = 0;
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = ubuf(partner[j]).d;
      buf[m++] = probability[j];
    }
    return m;
  }

 // topo handles special comm 
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = ubuf(finalpartner[j]).d;
    ns = nspecial[j][0];
    buf[m++] = ubuf(ns).d;
    for (k = 0; k < ns; k++)
      buf[m++] = ubuf(special[j][k]).d;
  }

  return m;
}

/* ---------------------------------------------------------------------- */

void FixBondBreakRxn::unpack_forward_comm(int n, int first, double *buf)
{
  int i,j,m,ns,last;

  if (commflag == 1) {
    m = 0;
    last = first + n;
    for (i = first; i < last; i++) {
      partner[i] = (tagint) ubuf(buf[m++]).i;
      probability[i] = buf[m++];
    }

  } else {

    int **nspecial = atom->nspecial;
    tagint **special = atom->special;

    m = 0;
    last = first + n;
    for (i = first; i < last; i++) {
      finalpartner[i] = (tagint) ubuf(buf[m++]).i;
      ns = (int) ubuf(buf[m++]).i;
      nspecial[i][0] = ns;
      for (j = 0; j < ns; j++)
        special[i][j] = (tagint) ubuf(buf[m++]).i;
    }
  }
}

/* ---------------------------------------------------------------------- */

int FixBondBreakRxn::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = ubuf(partner[i]).d;
    buf[m++] = distsq[i];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixBondBreakRxn::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    if (buf[m+1] > distsq[j]) {
      partner[j] = (tagint) ubuf(buf[m++]).i;
      distsq[j] = buf[m++];
    } else m += 2;
  }
}


/* ---------------------------------------------------------------------- */

void FixBondBreakRxn::print_bb()
{
  for (int i = 0; i < atom->nlocal; i++) {
    printf("TAG " TAGINT_FORMAT ": %d nbonds: ",atom->tag[i],atom->num_bond[i]);
    for (int j = 0; j < atom->num_bond[i]; j++) {
      printf(" %d",atom->bond_atom[i][j]);
    }
    printf("\n");
    printf("TAG " TAGINT_FORMAT ": %d nangles: ",atom->tag[i],atom->num_angle[i]);
    for (int j = 0; j < atom->num_angle[i]; j++) {
      printf(" %d %d %d,",atom->angle_atom1[i][j],
	     atom->angle_atom2[i][j],atom->angle_atom3[i][j]);
    }
    printf("\n");
    printf("TAG " TAGINT_FORMAT ": %d ndihedrals: ",atom->tag[i],atom->num_dihedral[i]);
    for (int j = 0; j < atom->num_dihedral[i]; j++) {
      printf(" %d %d %d %d,",atom->dihedral_atom1[i][j],
	     atom->dihedral_atom2[i][j],atom->dihedral_atom3[i][j],
	     atom->dihedral_atom4[i][j]);
    }
    printf("\n");
    printf("TAG " TAGINT_FORMAT ": %d %d %d nspecial: ",atom->tag[i],
	   atom->nspecial[i][0],atom->nspecial[i][1],atom->nspecial[i][2]);
    for (int j = 0; j < atom->nspecial[i][2]; j++) {
      printf(" %d",atom->special[i][j]);
    }
    printf("\n");
  }
}

/* ---------------------------------------------------------------------- */

void FixBondBreakRxn::print_copy(const char *str, tagint m, 
                              int n1, int n2, int n3, int *v)
{
  printf("%s %i: %d %d %d nspecial: ",str,m,n1,n2,n3);
  for (int j = 0; j < n3; j++) printf(" %d",v[j]);
  printf("\n");
}

/* ---------------------------------------------------------------------- */

double FixBondBreakRxn::compute_vector(int n)
{
  if (n == 0) return (double) breakcount;
  return (double) breakcounttotal;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixBondBreakRxn::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = 2*nmax * sizeof(tagint);
  bytes += nmax * sizeof(double);
  return bytes;
}
