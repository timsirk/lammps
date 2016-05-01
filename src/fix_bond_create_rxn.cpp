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
#include "fix_bond_create_rxn.h"
#include "update.h"
#include "respa.h"
#include "atom.h"
#include "atom_vec.h"
#include "force.h"
#include "pair.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "random_mars.h"
#include "memory.h"
#include "error.h"
#include "molecule.h"
#include "topo.h"

using namespace LAMMPS_NS;
using namespace FixConst;

#define BIG 1.0e20
#define DELTA 16

/* ---------------------------------------------------------------------- */

FixBondCreateRxn::FixBondCreateRxn(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 7) error->all(FLERR,"Illegal fix bond/create command");

  MPI_Comm_rank(world,&me);

  nevery = force->inumeric(FLERR,arg[3]);
  if (nevery <= 0) error->all(FLERR,"Illegal fix bond/create command");

  topoflag = 0;
  force_reneighbor = 1;
  next_reneighbor = -1;
  vector_flag = 1;
  size_vector = 2;
  global_freq = 1;
  extvector = 0;

  iatomtype = force->inumeric(FLERR,arg[4]);
  jatomtype = force->inumeric(FLERR,arg[5]);
  double cutoff = force->numeric(FLERR,arg[6]);

  if (iatomtype < 1 || iatomtype > atom->ntypes ||
      jatomtype < 1 || jatomtype > atom->ntypes)
    error->all(FLERR,"Invalid atom type in fix bond/create command");
  if (cutoff < 0.0) error->all(FLERR,"Illegal fix bond/create command");

  cutsq = cutoff*cutoff;

  // optional keywords

  imaxbond = 0;
  jmaxbond = 0;
  btype = 0;
  fraction = 1.0;
  int seed = 12345;

  int iarg = 7;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"imax") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix bond/create command");
      imaxbond = force->inumeric(FLERR,arg[iarg+1]);
      if (imaxbond < 0) error->all(FLERR,"Illegal fix bond/create command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"jmax") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix bond/create command");
      jmaxbond = force->inumeric(FLERR,arg[iarg+1]);
      if (jmaxbond < 0) error->all(FLERR,"Illegal fix bond/create command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"btype") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix bond/create command");
      btype = force->inumeric(FLERR,arg[iarg+1]);
      if (btype < 0 || btype > atom->nbondtypes) error->all(FLERR,"Illegal fix bond/create command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"prob") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix bond/create command");
      fraction = force->numeric(FLERR,arg[iarg+1]);
      seed = force->inumeric(FLERR,arg[iarg+2]);
      if (fraction < 0.0 || fraction > 1.0)
        error->all(FLERR,"Illegal fix bond/create command");
      if (seed <= 0) error->all(FLERR,"Illegal fix bond/create command");
      iarg += 3;
    } 
      else if (strcmp(arg[iarg],"mol") == 0) {
      int imol = atom->find_molecule(arg[iarg+1]);
      if (imol == -1) error->all(FLERR,"Molecule template ID for "
                                 "create_atoms does not exist");
      topoflag = 1;
      onemol = atom->molecules[imol];
      iarg += 2;
    } 
      else error->all(FLERR,"Illegal fix bond/create command");
  }

  // error check

  if (!topoflag)
    error->all(FLERR,"No rxn template was found");
  if (atom->molecular != 1)
    error->all(FLERR,"Cannot use fix bond/create with non-molecular systems");

  // initialize Marsaglia RNG with processor-unique seed

  random = new RanMars(lmp,seed + me);
  topo = new Topo(lmp);

  // perform initial allocation of atom-based arrays
  // register with Atom class
  // bondcount values will be initialized in setup()

  bondcount = NULL;
  grow_arrays(atom->nmax);
  atom->add_callback(0);
  countflag = 0;

  // set comm sizes needed by this fix
  // topo class does the special comm 
  // still need small forward and reverse comm 

  comm_forward = 2;
  comm_reverse = 2;

  // allocate arrays local to this fix

  nmax = 0;
  partner = finalpartner = NULL;
  distsq = NULL;

  maxcreate = 0;
  created = NULL;

  // zero out stats
  createcount = 0;
  createcounttotal = 0;
}

/* ---------------------------------------------------------------------- */

FixBondCreateRxn::~FixBondCreateRxn()
{
  // unregister callbacks to this fix from Atom class

  atom->delete_callback(id,0);

  delete random;
  delete topo;

  // delete locally stored arrays

  memory->destroy(bondcount);
  memory->destroy(partner);
  memory->destroy(finalpartner);
  memory->destroy(distsq);
  memory->destroy(created);
}

/* ---------------------------------------------------------------------- */

int FixBondCreateRxn::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= POST_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixBondCreateRxn::init()
{
  if (strstr(update->integrate_style,"respa"))
    nlevels_respa = ((Respa *) update->integrate)->nlevels;

  // check cutoff for iatomtype,jatomtype

  if (force->pair == NULL || cutsq > force->pair->cutsq[iatomtype][jatomtype])
    error->all(FLERR,"Fix bond/create cutoff is longer than pairwise cutoff");
  if (force->improper) {
    if (force->improper_match("class2") || force->improper_match("ring"))
      error->all(FLERR,"Cannot yet use fix bond/create with this "
                 "improper style");
  }

  // need a half neighbor list, built every Nevery steps
  int irequest = neighbor->request((void *) this);

  //int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->fix = 1;
  neighbor->requests[irequest]->occasional = 1;

  lastcheck = -1;
}

/* ---------------------------------------------------------------------- */

void FixBondCreateRxn::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixBondCreateRxn::setup(int vflag)
{
  int i,j,m;

  // compute initial bondcount if this is first run
  // can't do this earlier, in constructor or init, b/c need ghost info

  if (countflag) return;
  countflag = 1;

  // count bonds stored with each bond I own
  // if newton bond is not set, just increment count on atom I
  // if newton bond is set, also increment count on atom J even if ghost
  // bondcount is long enough to tally ghost atom counts

  int *num_bond = atom->num_bond;
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;
  int newton_bond = force->newton_bond;

  for (i = 0; i < nall; i++) bondcount[i] = 0;

  for (i = 0; i < nlocal; i++)
    for (j = 0; j < num_bond[i]; j++) {
      if (bond_type[i][j] == btype || btype == 0) {
        bondcount[i]++;
        if (newton_bond) {
          m = atom->map(bond_atom[i][j]);
          if (m < 0) 
            error->one(FLERR,"Fix bond/create needs ghost atoms "
                       "from further away");
          bondcount[m]++;
        }
      }
    }

  // if newton_bond is set, need to sum bondcount

  commflag = 1;
  if (newton_bond) comm->reverse_comm_fix(this,1);
}

/* ---------------------------------------------------------------------- */

void FixBondCreateRxn::post_integrate()
{
  int i,j,k,m,n,ii,jj,inum,jnum,itype,jtype,n1,n2,n3,possible;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  int *ilist,*jlist,*numneigh,**firstneigh;
  tagint *slist;

  if (update->ntimestep % nevery) return;

  // check that all procs have needed ghost atoms within ghost cutoff
  // only if neighbor list has changed since last check
  // needs to be <= test b/c neighbor list could have been re-built in
  //   same timestep as last post_integrate() call, but afterwards
  // NOTE: no longer think is needed, due to error tests on atom->map()
  // NOTE: if delete, can also delete lastcheck and check_ghosts()

  if (lastcheck <= neighbor->lastcall) check_ghosts();

  // acquire updated ghost atom positions
  // necessary b/c are calling this after integrate, but before Verlet comm

  comm->forward_comm();

  // forward comm of bondcount, so ghosts have it

  commflag = 1;
  comm->forward_comm_fix(this,1);

  // resize bond partner list and initialize it
  // probability array overlays distsq array
  // needs to be atom->nmax in length

  if (atom->nmax > nmax) {
    memory->destroy(partner);
    memory->destroy(finalpartner);
    memory->destroy(distsq);
    nmax = atom->nmax;
    memory->create(partner,nmax,"bond/create:partner");
    memory->create(finalpartner,nmax,"bond/create:finalpartner");
    memory->create(distsq,nmax,"bond/create:distsq");
    probability = distsq;
  }

  int nlocal = atom->nlocal;
  int nall = atom->nlocal + atom->nghost;

  for (i = 0; i < nall; i++) {
    partner[i] = 0;
    finalpartner[i] = 0;
    distsq[i] = BIG;
  }

  next_reneighbor = -1;

  // loop over neighbors of my atoms
  // each atom sets one closest eligible partner atom ID to bond with

  double **x = atom->x;
  tagint *tag = atom->tag;
  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int *mask = atom->mask;
  int *type = atom->type;

  neighbor->build_one(list,1);
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    if (!(mask[i] & groupbit)) continue;
    itype = type[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      if (!(mask[j] & groupbit)) continue;
      jtype = type[j];

      possible = 0;
      if (itype == iatomtype && jtype == jatomtype) {
        if ((imaxbond == 0 || bondcount[i] < imaxbond) &&
            (jmaxbond == 0 || bondcount[j] < jmaxbond))
          possible = 1;
      } else if (itype == jatomtype && jtype == iatomtype) {
        if ((jmaxbond == 0 || bondcount[i] < jmaxbond) &&
            (imaxbond == 0 || bondcount[j] < imaxbond))
          possible = 1;
      }
      if (!possible) continue;

      // do not allow a duplicate bond to be created
      // check 1-2 neighbors of atom I

      for (k = 0; k < nspecial[i][0]; k++)
        if (special[i][k] == tag[j]) possible = 0;
      if (!possible) continue;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      if (rsq >= cutsq) continue;

      if (rsq < distsq[i]) {
        partner[i] = tag[j];
        distsq[i] = rsq;
      }
      if (rsq < distsq[j]) {
        partner[j] = tag[i];
        distsq[j] = rsq;
      }
    }
  }

  // reverse comm of distsq and partner
  // not needed if newton_pair off since I,J pair was seen by both procs
  commflag = 2;
  if (force->newton_pair) comm->reverse_comm_fix(this);

  // each atom now knows its winning partner
  // for prob check, generate random value for each atom with a bond partner
  // forward comm of partner and random value, so ghosts have it
  if (fraction < 1.0) {
    for (i = 0; i < nlocal; i++)
      if (partner[i]) probability[i] = random->uniform();
  }

  commflag = 2;
  comm->forward_comm_fix(this,2);

  // create bonds for atoms I own
  // only if both atoms list each other as winning bond partner
  //   and probability constraint is satisfied
  // if other atom is owned by another proc, it should do same thing
  int **bond_type = atom->bond_type;
  int newton_bond = force->newton_bond;

  ncreate = 0;
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

    // atom J will also do this, whatever proc it is on
    bondcount[i]++;

    // store final created bond partners and count the created bond once
    finalpartner[i] = tag[j];
    finalpartner[j] = tag[i];
    if (tag[i] < tag[j]) ncreate++;
  }

  // tally stats
  MPI_Allreduce(&ncreate,&createcount,1,MPI_INT,MPI_SUM,world);
  createcounttotal += createcount;

  // trigger reneighboring if any bonds were formed
  // this insures neigh lists will immediately reflect the topology changes
  // done if any bonds created
  if (createcount) next_reneighbor = update->ntimestep;
  if (!createcount) return;

  // communicate final partner and 1-2 special neighbors
  // 1-2 neighs already reflect created bonds
  commflag = 3;
  comm->forward_comm_fix(this);

  // topo handles the rest
  // actually make the bonds
  // update topology and influenced atoms
    topo->change_bonds(1, finalpartner, onemol);
    return;
}

/* ----------------------------------------------------------------------
   insure all atoms 2 hops away from owned atoms are in ghost list
   this allows dihedral 1-2-3-4 to be properly created
     and special list of 1 to be properly updated
   if I own atom 1, but not 2,3,4, and bond 3-4 is added
     then 2,3 will be ghosts and 3 will store 4 as its finalpartner
------------------------------------------------------------------------- */

void FixBondCreateRxn::check_ghosts()
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
    error->all(FLERR,"Fix bond/create needs ghost atoms from further away");
  lastcheck = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

void FixBondCreateRxn::post_integrate_respa(int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) post_integrate();
}

/* ---------------------------------------------------------------------- */

int FixBondCreateRxn::pack_forward_comm(int n, int *list, double *buf,
                                     int pbc_flag, int *pbc)
{
  int i,j,k,m,ns;

  m = 0;

  if (commflag == 1) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = ubuf(bondcount[j]).d;
    }
    return m;
  }

  if (commflag == 2) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = ubuf(partner[j]).d;
      buf[m++] = probability[j];
    }
    return m;
  }

 // topo handles special comm if present
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = ubuf(finalpartner[j]).d;
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixBondCreateRxn::unpack_forward_comm(int n, int first, double *buf)
{
  int i,j,m,ns,last;

  m = 0;
  last = first + n;

  if (commflag == 1) {
    for (i = first; i < last; i++)
      bondcount[i] = (int) ubuf(buf[m++]).i;

  } else if (commflag == 2) {
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
    }
  }
}

/* ---------------------------------------------------------------------- */

int FixBondCreateRxn::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;

  if (commflag == 1) {
    for (i = first; i < last; i++)
      buf[m++] = ubuf(bondcount[i]).d;
    return m;
  }

  for (i = first; i < last; i++) {
    buf[m++] = ubuf(partner[i]).d;
    buf[m++] = distsq[i];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixBondCreateRxn::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;

  if (commflag == 1) {
    for (i = 0; i < n; i++) {
      j = list[i];
      bondcount[j] += (int) ubuf(buf[m++]).i;
    }

  } else {
    for (i = 0; i < n; i++) {
      j = list[i];
      if (buf[m+1] < distsq[j]) {
        partner[j] = (tagint) ubuf(buf[m++]).i;
        distsq[j] = buf[m++];
      } else m += 2;
    }
  }
}

/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */

void FixBondCreateRxn::grow_arrays(int nmax)
{
  memory->grow(bondcount,nmax,"bond/create:bondcount");
}

/* ----------------------------------------------------------------------
   copy values within local atom-based arrays
------------------------------------------------------------------------- */

void FixBondCreateRxn::copy_arrays(int i, int j, int delflag)
{
  bondcount[j] = bondcount[i];
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for exchange with another proc
------------------------------------------------------------------------- */

int FixBondCreateRxn::pack_exchange(int i, double *buf)
{
  buf[0] = bondcount[i];
  return 1;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based arrays from exchange with another proc
------------------------------------------------------------------------- */

int FixBondCreateRxn::unpack_exchange(int nlocal, double *buf)
{
  bondcount[nlocal] = static_cast<int> (buf[0]);
  return 1;
}

/* ---------------------------------------------------------------------- */

double FixBondCreateRxn::compute_vector(int n)
{
  if (n == 0) return (double) createcount;
  return (double) createcounttotal;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixBondCreateRxn::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = nmax * sizeof(int);
  bytes = 2*nmax * sizeof(tagint);
  bytes += nmax * sizeof(double);
  return bytes;
}

/* ---------------------------------------------------------------------- */

void FixBondCreateRxn::print_bb()
{
  for (int i = 0; i < atom->nlocal; i++) {
    printf("TAG " TAGINT_FORMAT ": %d nbonds: ",atom->tag[i],atom->num_bond[i]);
    for (int j = 0; j < atom->num_bond[i]; j++) {
      printf(" " TAGINT_FORMAT,atom->bond_atom[i][j]);
    }
    printf("\n");
    printf("TAG " TAGINT_FORMAT ": %d nangles: ",atom->tag[i],atom->num_angle[i]);
    for (int j = 0; j < atom->num_angle[i]; j++) {
      printf(" " TAGINT_FORMAT " " TAGINT_FORMAT " " TAGINT_FORMAT ",",
             atom->angle_atom1[i][j], atom->angle_atom2[i][j],
             atom->angle_atom3[i][j]);
    }
    printf("\n");
    printf("TAG " TAGINT_FORMAT ": %d ndihedrals: ",atom->tag[i],atom->num_dihedral[i]);
    for (int j = 0; j < atom->num_dihedral[i]; j++) {
      printf(" " TAGINT_FORMAT " " TAGINT_FORMAT " " TAGINT_FORMAT " " 
             TAGINT_FORMAT ",", atom->dihedral_atom1[i][j],
	     atom->dihedral_atom2[i][j],atom->dihedral_atom3[i][j],
	     atom->dihedral_atom4[i][j]);
    }
    printf("\n");
    printf("TAG " TAGINT_FORMAT ": %d nimpropers: ",atom->tag[i],atom->num_improper[i]);
    for (int j = 0; j < atom->num_improper[i]; j++) {
      printf(" " TAGINT_FORMAT " " TAGINT_FORMAT " " TAGINT_FORMAT " " 
             TAGINT_FORMAT ",",atom->improper_atom1[i][j],
	     atom->improper_atom2[i][j],atom->improper_atom3[i][j],
	     atom->improper_atom4[i][j]);
    }
    printf("\n");
    printf("TAG " TAGINT_FORMAT ": %d %d %d nspecial: ",atom->tag[i],
	   atom->nspecial[i][0],atom->nspecial[i][1],atom->nspecial[i][2]);
    for (int j = 0; j < atom->nspecial[i][2]; j++) {
      printf(" " TAGINT_FORMAT,atom->special[i][j]);
    }
    printf("\n");
  }
}

/* ---------------------------------------------------------------------- */

void FixBondCreateRxn::print_copy(const char *str, tagint m, 
                              int n1, int n2, int n3, int *v)
{
  printf("%s " TAGINT_FORMAT ": %d %d %d nspecial: ",str,m,n1,n2,n3);
  for (int j = 0; j < n3; j++) printf(" %d",v[j]);
  printf("\n");
}
