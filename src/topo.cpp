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

#include "stdio.h"
#include "math.h"
#include "stdlib.h"
#include "string.h"
#include "atom.h"
#include "force.h"
#include "bond.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "group.h"
#include "comm.h"
#include "domain.h"
#include "modify.h"
#include "compute.h"
#include "random_mars.h"
#include "citeme.h"
#include "memory.h"
#include "error.h"
#include "update.h"
#include "special.h"
#include "topo.h"
#include "molecule.h"
#include "fix.h"
#include "atom_vec.h"

using namespace LAMMPS_NS;

Topo *Topo::cptr;

#define DELTA 16
#define MAX_DUP_DIH 10
#define EPSILON 0.001

/* ---------------------------------------------------------------------- */
Topo::Topo(LAMMPS *lmp) : Pointers(lmp) {

  // this arr/y not atom-based, grow later as needed
  influencedlist = NULL;
  maxinfluenced = DELTA;
  maxcreate = 0;
  memory->grow(influencedlist,maxinfluenced,"topo:influencedlist");

  // per-atom array to comm special list
  test = NULL;
  maxnmax = atom->nmax;
  memory->grow(test,maxnmax,2+atom->maxspecial,"topo:test");

  // forward comm to update group info
  //comm_forward = MAX(2,2+atom->maxspecial);

  memory->create(created,DELTA,2,"topo:created");
  // copy = special list for one atom
  // size = ms^2 + ms is sufficient
  // b/c in rebuild_special() neighs of all 1-2s are added,
  //   then a dedup(), then neighs of all 1-3s are added, then final dedup()
  // this means intermediate size cannot exceed ms^2 + ms

  int maxspecial = atom->maxspecial;
  copy = new tagint[maxspecial*maxspecial + maxspecial];

  // future could add hash for reaction atom types and reaction charges
  // instead of listing each type in the rxn template
}
/* ---------------------------------------------------------------------- */
Topo::~Topo(){

  if(influencedlist)
    memory->destroy(influencedlist);
  delete [] copy;
}
/* ----------------------------------------------------------------------
dump the bonds as debug
------------------------------------------------------------------------- */
int Topo::printBonds(bigint count, int** bond)
{

created = bond;
 
  for(int i = 0; i < count; i++)
    printf("xcount %lld bond %d %d \n", count, created[i][0], created[i][1]);

  return 0;
}

/* ----------------------------------------------------------------------
update the bonds, angles, dih, impr, types, charges 
------------------------------------------------------------------------- */
int Topo::change_bonds(int bondflag, int* finalpartnerfromfix, Molecule *mymol)
{

  // flag 0 delete bonds
  // flag 1 create bonds
  flag = bondflag;

  MPI_Comm_rank(world,&me);
  double **x = atom->x; 
  tagint *tag = atom->tag;
  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int *mask = atom->mask;
  int *type = atom->type;

  // local ptrs to atom arrays 
  tagint *molecule = atom->molecule; 
  int **bond_type = atom->bond_type; 
  int *num_angle = atom->num_angle;
  int newton_bond = force->newton_bond; 
  int nlocal = atom->nlocal;  
  int nall = nlocal + atom->nghost;
//  int createcount = 0;

  // mymol ptr can be passed in by various fixes, etc
  onemol = mymol;
  finalpartner = finalpartnerfromfix;
  // working variables
  int i,j, tmp;

// call back for per atom arrays (special bonds)
//memory->grow(test,atom->nmax,"topo:");
//atom->add_callback(0);

  // allocate and initialize deletion list
  memory->create(dlist,nlocal,"topo:dlist");

 // settings
    if (onemol->nrangles > 0)
      angleflag = 1;
    else 
      angleflag = 0;
    if (onemol->nrdihedrals > 0)
      dihedralflag = 1;
    else 
      dihedralflag = 0;
    if (onemol->nrimpropers > 0)
      improperflag = 1;
    else 
      improperflag = 0;

   // working variables for speical bond update
    tagint *slist;
    int n,n1,n2,n3,m;

  // create bonds
  if(flag){

    ncreatelocal = 0; 
    for (i = 0; i < nlocal; i++) {
      if (finalpartner[i] == 0) continue;
      j = atom->map(finalpartner[i]);
      if (finalpartner[j] != tag[i]) continue;

      // if newton_bond is set, only store with I or J
      // if not newton_bond, store bond with both I and J
      // atom J will also do this consistently, whatever proc it is on

      if (!newton_bond || tag[i] < tag[j]) {
        if (num_bond[i] == atom->bond_per_atom)
          error->all(FLERR,"New bond exceeded bonds per atom in topo command");

      btype = check_btype(i,j);
      if(!btype)
        continue; 

      bond_type[i][num_bond[i]] = btype;
      bond_atom[i][num_bond[i]] = tag[j];
      num_bond[i]++;
    }    

    // add a 1-2 neighbor to special bond list for atom I
    // atom J will also do this, whatever proc it is on
    // need to first remove tag[j] from later in list if it appears
    // prevents list from overflowing, will be rebuilt in rebuild_special()

    slist = special[i];
    n1 = nspecial[i][0];
    n2 = nspecial[i][1];
    n3 = nspecial[i][2];
    for (m = n1; m < n3; m++) 
      if (slist[m] == tag[j]) break;
    if (m < n3) {
      for (n = m; n < n3-1; n++) slist[n] = slist[n+1];
      n3--;
      if (m < n2) n2--;
    }    
    if (n3 == atom->maxspecial)
      error->one(FLERR,
                 "New bond exceeded special list size in topo command");
    for (m = n3; m > n1; m--) slist[m] = slist[m-1];
    slist[n1] = tag[j];
    nspecial[i][0] = n1+1;
    nspecial[i][1] = n2+1;
    nspecial[i][2] = n3+1;

    // count the created bond once

    if (tag[i] < tag[j]) ncreatelocal++;
   }
  }

  // delete bonds
  if(!flag){

  ncreatelocal = 0; 
  nbreaklocal = 0;
  for (i = 0; i < nlocal; i++) {
    if (finalpartner[i] == 0) continue;
    j = atom->map(finalpartner[i]);
    if (finalpartner[j] != tag[i]) continue;

    // delete bond from atom I if I stores it
    // atom J will also do this

    for (m = 0; m < num_bond[i]; m++) {
      if (bond_atom[i][m] == finalpartner[i]) {
        for (int k = m; k < num_bond[i]-1; k++) {
          bond_atom[i][k] = bond_atom[i][k+1];
          bond_type[i][k] = bond_type[i][k+1];
        }   
        num_bond[i]--;
        break;
      }   
    }   

    // remove J from special bond list for atom I
    // atom J will also do this, whatever proc it is on

    slist = special[i];
    n1 = nspecial[i][0];
    for (m = 0; m < n1; m++)
      if (slist[m] == finalpartner[i]) break;
    n3 = nspecial[i][2];
    for (; m < n3-1; m++) slist[m] = slist[m+1];
    nspecial[i][0]--;
    nspecial[i][1]--;
    nspecial[i][2]--;

    // count the broken bond once

    if (tag[i] < tag[j]) ncreatelocal++;
  }
 }
  // make created list
  // this is list of new bonds that influence my owned atoms
  //   even if between owned-ghost or ghost-ghost atoms
  // finalpartner was passed in for owned and ghost atoms so loop over nall
  // OK if duplicates in broken list due to ghosts duplicating owned atoms
  // check J < 0 to insure a broken bond to unknown atom is included
  //   i.e. a bond partner outside of cutoff length

  //memory->create(created,DELTA,2,"topo:created");

  ncreate = 0;
  for (i = 0; i < nall; i++) {
    if (finalpartner[i] == 0) continue;
    j = atom->map(finalpartner[i]);

    btype = check_btype(i,j);
    if(!btype)
      continue; 

    if (j < 0 || tag[i] < tag[j]) {
      if (ncreate == maxcreate) {
        maxcreate += DELTA;
        memory->grow(created,maxcreate,2,"topo:created");
      }
      created[ncreate][0] = tag[i];
      created[ncreate][1] = finalpartner[i];
      ncreate++;
    }
  }

  // comm of special lists
  // need only 1-2 neighs for ghosts
  comm_special();

  update_topology();

  // delete if type -1
  //delete_atom();

  comm->forward_comm();

  return 0;
}

/* ----------------------------------------------------------------------
   double loop over my atoms and created bonds
   influenced = 1 if atom's topology is affected by any created bond
     yes if is one of 2 atoms in bond
     yes if either atom ID appears in as 1-2 or 1-3 in atom's special list
     else no
   if influenced by any created bond:
     rebuild the atom's special list of 1-2,1-3,1-4 neighs
     check for angles/dihedrals/impropers to create due modified special list
------------------------------------------------------------------------- */

void Topo::update_topology()
{
  int i,j,k,n,influence,influenced,found;
  int specialflag12;
  tagint id1,id2;
  tagint *slist,*slistii;

  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
//  tagint *type = atom->type;
  int nlocal = atom->nlocal;

  nangles = 0;
  ndihedrals = 0;
  nimpropers = 0;
  overflow = 0;

  int ninfluenced = 0;

  // set flag to compute 1-2 1-2 bond neighs if special bonds 011 or 111
  if (fabs(force->special_lj[3] - 1.0) < EPSILON || fabs(force->special_coul[3] - 1.0) < EPSILON)
    specialflag12 = 1;
  else if (fabs(force->special_lj[2] - 1.0) < EPSILON || fabs(force->special_coul[2] - 1.0) < EPSILON)
    specialflag12 = 1;
  else 
   specialflag12 = 0;

  // see if each atom i is influenced by new bond
  int ii, nn, kk;
  for (i = 0; i < nlocal; i++) {
    influence = 0;
    slist = special[i];

    for (j = 0; j < ncreate; j++) {
      id1 = created[j][0];
      id2 = created[j][1];
      influence = 0;
      if (tag[i] == id1 || tag[i] == id2){
        influence = 1;
        goto done;
      }
      else {
        // check 1-2 bond neighs and 1-3, 1-4 if present
        n = nspecial[i][1];
        for (k = 0; k < n; k++){
          if (slist[k] == id1 || slist[k] == id2) {
            influence = 1;
            goto done;
          }
          // also need to check 1-2 1-2 neighs if special is 011, 111
          if(specialflag12){

            ii = atom->map(slist[k]);

            if (ii < 0)
              error->one(FLERR,"Topo command needs ghost atoms from further away");
            nn = nspecial[ii][0];
            slistii = special[ii];

            for (kk = 0; kk < nn; kk++){
              if (slistii[kk] == id1 || slistii[kk] == id2) {
                influence = 1;
              }
              if(influence) 
                goto done;
            }
          }
        }
      }
    }

done:

   // rebuild_special first, since used by create_angles, etc
    if (influence) {

      // delete atom and its topology
      // i is local to this proc 
      //if(onemol->rtype[atom->type[i]-1] == -1){
      //  dlist[i] = 1;
      //  continue;
      //}

        rebuild_special(i);

      if (angleflag){ 
        if(flag) 
          create_angles(i);
        else 
          break_angles(i,id1,id2);
      }
      if (dihedralflag){
        if(flag) 
          create_dihedrals(i);
        else 
          break_dihedrals(i,id1,id2);
      }
      if (improperflag){
        if(flag) 
          create_impropers(i);
        else 
          break_impropers(i,id1,id2);
      }

        influencedlist[ninfluenced++] = i;
        if(ninfluenced == maxinfluenced){  
          maxinfluenced += DELTA;
          memory->grow(influencedlist, maxinfluenced, "topo:influencedlist");
        }
    }
  }

  // update atom types for influenced atoms 
  // must be done after angle, dih, impro are matched

    for (int ii = 0; ii < ninfluenced; ii++) {

      i = influencedlist[ii];
      // charge update
      if (atom->q_flag)
        atom->q[i] = onemol->rq[atom->type[i]-1];
      // type update
      atom->type[i] = onemol->rtype[atom->type[i]-1];
    }

  // delete any atoms with type -1
  //      delete_atom();

  int overflowall;
  int newton_bond = force->newton_bond;
  int all;

  MPI_Allreduce(&overflow,&overflowall,1,MPI_INT,MPI_SUM,world);

  if (overflowall) error->all(FLERR,"Topo command induced too many "
                              "angles/dihedrals/impropers per atom");

  // update nbonds, nangles, etc
  MPI_Allreduce(&ncreatelocal,&all,1,MPI_INT,MPI_SUM,world);
  if (flag)
    atom->nbonds += all;
  else 
    atom->nbonds -= all;

  if (angleflag) {
    MPI_Allreduce(&nangles,&all,1,MPI_INT,MPI_SUM,world);
    if (!newton_bond) all /= 3;
    if (flag)
      atom->nangles += all;
    else 
      atom->nangles -= all;
  }
  if (dihedralflag) {
    MPI_Allreduce(&ndihedrals,&all,1,MPI_INT,MPI_SUM,world);
    if (!newton_bond) all /= 4;
    if (flag)
      atom->ndihedrals += all;
    else
      atom->ndihedrals -= all;
  }
  if (improperflag) {
    MPI_Allreduce(&nimpropers,&all,1,MPI_INT,MPI_SUM,world);
    if (!newton_bond) all /= 4;
    if (flag)
      atom->nimpropers += all;
    else
      atom->nimpropers -= all;
  }
}

/* ----------------------------------------------------------------------
   re-build special list of atom M
   does not affect 1-2 neighs (already include effects of new bond)
   affects 1-3 and 1-4 neighs due to other atom's augmented 1-2 neighs
------------------------------------------------------------------------- */

void Topo::rebuild_special(int m)
{
  int i,j,n,n1,cn1,cn2,cn3;
  tagint *slist;

  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  // existing 1-2 neighs of atom M
  slist = special[m];
  n1 = nspecial[m][0];

  cn1 = 0;

  for (i = 0; i < n1; i++)
    copy[cn1++] = slist[i];

  // new 1-3 neighs of atom M, based on 1-2 neighs of 1-2 neighs
  // exclude self
  // remove duplicates after adding all possible 1-3 neighs

  cn2 = cn1;

  for (i = 0; i < cn1; i++) {
    n = atom->map(copy[i]);
    if (n < 0) 
      error->one(FLERR,"Topo command needs ghost atoms from further away");
    slist = special[n];
    n1 = nspecial[n][0];
    for (j = 0; j < n1; j++)
      if (slist[j] != tag[m]) copy[cn2++] = slist[j];
  }

  cn2 = dedup(cn1,cn2,copy);
  if (cn2 > atom->maxspecial)
    error->one(FLERR,"Special list size exceeded in Topo command");

  // new 1-4 neighs of atom M, based on 1-2 neighs of 1-3 neighs
  // exclude self
  // remove duplicates after adding all possible 1-4 neighs
  cn3 = cn2;

  for (i = cn1; i < cn2; i++) {
    n = atom->map(copy[i]);
    if (n < 0) 
      error->one(FLERR,"Topo command needs ghost atoms from further away");
    slist = special[n];
    n1 = nspecial[n][0];
    for (j = 0; j < n1; j++)
      if (slist[j] != tag[m]) copy[cn3++] = slist[j];
  }

  cn3 = dedup(cn2,cn3,copy);
  if (cn3 > atom->maxspecial)
    error->one(FLERR,"Special list size exceeded in Topo command");

  // store new special list with atom M
  nspecial[m][0] = cn1;
  nspecial[m][1] = cn2;
  nspecial[m][2] = cn3;
  memcpy(special[m],copy,cn3*sizeof(int));
}

/* ----------------------------------------------------------------------
   create any angles owned by atom M induced by newly created bonds
   walk special list to find all possible angles to create
   only add an angle if a new bond is one of its 2 bonds (I-J,J-K)
   for newton_bond on, atom M is central atom
   for newton_bond off, atom M is any of 3 atoms in angle
------------------------------------------------------------------------- */

void Topo::create_angles(int m)
{
  int i,j,n,i2local,n1,n2;
  tagint i1,i2,i3;
  tagint *s1list,*s2list;

  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  int* type = atom->type;
  int num_angle = atom->num_angle[m];
  int *angle_type = atom->angle_type[m];
  tagint *angle_atom1 = atom->angle_atom1[m];
  tagint *angle_atom2 = atom->angle_atom2[m];
  tagint *angle_atom3 = atom->angle_atom3[m];

  // atom M is central atom in angle
  // double loop over 1-2 neighs
  // avoid double counting by 2nd loop as j = i+1,N not j = 1,N
  // consider all angles, only add if:
  //   a new bond is in the angle and atom types match

  i2 = tag[m];
  n2 = nspecial[m][0];
  s2list = special[m];

  for (i = 0; i < n2; i++) {
    i1 = s2list[i];
    for (j = i+1; j < n2; j++) {
      i3 = s2list[j];

      // angle = i1-i2-i3

      for (n = 0; n < ncreate; n++) {
        if (created[n][0] == i1 && created[n][1] == i2) break;
        if (created[n][0] == i2 && created[n][1] == i1) break;
        if (created[n][0] == i2 && created[n][1] == i3) break;
        if (created[n][0] == i3 && created[n][1] == i2) break;
      }
      if (n == ncreate) continue;

      // 0 if not present
      // angle number if present
        atype = check_atype(i1,i2,i3); 
        if(!atype) continue; 

      if (num_angle < atom->angle_per_atom) {
        angle_type[num_angle] = atype;
        angle_atom1[num_angle] = i1;
        angle_atom2[num_angle] = i2;
        angle_atom3[num_angle] = i3;
        num_angle++;
        nangles++;
      } else overflow = 1;
    }
  }

  atom->num_angle[m] = num_angle;
  if (force->newton_bond) return;

  // for newton_bond off, also consider atom M as atom 1 in angle

  i1 = tag[m];
  n1 = nspecial[m][0];
  s1list = special[m];

  for (i = 0; i < n1; i++) {
    i2 = s1list[i];
    i2local = atom->map(i2);
    s2list = special[i2local];
    n2 = nspecial[i2local][0];

    for (j = 0; j < n2; j++) {
      i3 = s2list[j];
      if (i3 == i1) continue;

      // angle = i1-i2-i3

      for (n = 0; n < ncreate; n++) {
        if (created[n][0] == i1 && created[n][1] == i2) break;
        if (created[n][0] == i2 && created[n][1] == i1) break;
        if (created[n][0] == i2 && created[n][1] == i3) break;
        if (created[n][0] == i3 && created[n][1] == i2) break;
      }
      if (n == ncreate) continue;

        atype = check_atype(i1,i2,i3); 
        if(!atype) continue; 

      if (num_angle < atom->angle_per_atom) {
        angle_type[num_angle] = atype;
        angle_atom1[num_angle] = i1;
        angle_atom2[num_angle] = i2;
        angle_atom3[num_angle] = i3;
        num_angle++;
        nangles++;
      } else overflow = 1;
    }
  }

  atom->num_angle[m] = num_angle;
}

/* ----------------------------------------------------------------------
   create any dihedrals owned by atom M induced by newly created bonds
   walk special list to find all possible dihedrals to create
   only add a dihedral if a new bond is one of its 3 bonds (I-J,J-K,K-L)
   for newton_bond on, atom M is central atom
   for newton_bond off, atom M is any of 4 atoms in dihedral
------------------------------------------------------------------------- */

void Topo::create_dihedrals(int m)
{
  int i,j,k,n,i1local,i2local,i3local,n1,n2,n3,n4;
  tagint i1,i2,i3,i4;
  tagint *s1list,*s2list,*s3list;

  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  int num_dihedral = atom->num_dihedral[m];
  int *dihedral_type = atom->dihedral_type[m];
  tagint *dihedral_atom1 = atom->dihedral_atom1[m];
  tagint *dihedral_atom2 = atom->dihedral_atom2[m];
  tagint *dihedral_atom3 = atom->dihedral_atom3[m];
  tagint *dihedral_atom4 = atom->dihedral_atom4[m];

  // atom M is 2nd atom in dihedral
  // double loop over 1-2 neighs
  // two triple loops: one over neighs at each end of triplet
  // avoid double counting by 2nd loop as j = i+1,N not j = 1,N
  // avoid double counting due to another atom being 2nd atom in same dihedral
  //   by requiring ID of 2nd atom < ID of 3rd atom
  //   don't do this if newton bond off since want to double count
  // consider all dihedrals, only add if:
  //   a new bond is in the dihedral and atom types match

  i2 = tag[m];
  n2 = nspecial[m][0];
  s2list = special[m];

  int dihedrals[MAX_DUP_DIH];
  int dcheck;

  for (i = 0; i < n2; i++) {
    i1 = s2list[i];

    for (j = i+1; j < n2; j++) {
      i3 = s2list[j];
      if (force->newton_bond && i2 > i3) continue;
      i3local = atom->map(i3);
      s3list = special[i3local];
      n3 = nspecial[i3local][0];

      for (k = 0; k < n3; k++) {
        i4 = s3list[k];
        if (i4 == i1 || i4 == i2 || i4 == i3) continue;

        // dihedral = i1-i2-i3-i4

        for (n = 0; n < ncreate; n++) {
          if (created[n][0] == i1 && created[n][1] == i2) break;
          if (created[n][0] == i2 && created[n][1] == i1) break;
          if (created[n][0] == i2 && created[n][1] == i3) break;
          if (created[n][0] == i3 && created[n][1] == i2) break;
          if (created[n][0] == i3 && created[n][1] == i4) break;
          if (created[n][0] == i4 && created[n][1] == i3) break;
        }
        if (n < ncreate) {
          // check types 
            dcheck = check_dtype(i1,i2,i3,i4,dihedrals); 
            if(!dcheck)
              continue; 

        // dihedrals zero index has number
        // followed by dihedral types
        // loop over dihedrals match in template, add them
        for(int zz = 1; zz <= dihedrals[0]; zz++){
          dtype = dihedrals[zz];
          if (num_dihedral < atom->dihedral_per_atom) {
            dihedral_type[num_dihedral] = dtype;
            dihedral_atom1[num_dihedral] = i1;
            dihedral_atom2[num_dihedral] = i2;
            dihedral_atom3[num_dihedral] = i3;
            dihedral_atom4[num_dihedral] = i4;
            num_dihedral++;
            ndihedrals++;
          } else overflow = 1; 
        }    
        }
      }
    }
  }

  for (i = 0; i < n2; i++) {
    i1 = s2list[i];
    if (force->newton_bond && i2 > i1) continue;
    i1local = atom->map(i1);
    s3list = special[i1local];
    n3 = nspecial[i1local][0];

    for (j = i+1; j < n2; j++) {
      i3 = s2list[j];

      for (k = 0; k < n3; k++) {
        i4 = s3list[k];
        if (i4 == i1 || i4 == i2 || i4 == i3) continue;

        // dihedral = i3-i2-i1-i4

        for (n = 0; n < ncreate; n++) {
          if (created[n][0] == i3 && created[n][1] == i2) break;
          if (created[n][0] == i2 && created[n][1] == i3) break;
          if (created[n][0] == i2 && created[n][1] == i1) break;
          if (created[n][0] == i1 && created[n][1] == i2) break;
          if (created[n][0] == i1 && created[n][1] == i4) break;
          if (created[n][0] == i4 && created[n][1] == i1) break;
        }
        if (n < ncreate) {

          // check types 
            dcheck = check_dtype(i3,i2,i1,i4,dihedrals); 
            if(!dcheck)
              continue; 

        // dihedrals zero index has number
        // followed by dihedral types
        // loop over dihedrals match in template, add them
        for(int zz = 1; zz <= dihedrals[0]; zz++){
          dtype = dihedrals[zz];
          if (num_dihedral < atom->dihedral_per_atom) {
            dihedral_type[num_dihedral] = dtype;
            dihedral_atom1[num_dihedral] = i3;
            dihedral_atom2[num_dihedral] = i2;
            dihedral_atom3[num_dihedral] = i1;
            dihedral_atom4[num_dihedral] = i4;
            num_dihedral++;
            ndihedrals++;
          } else overflow = 1; 
        }    

        }
      }
    }
  }

  atom->num_dihedral[m] = num_dihedral;
  if (force->newton_bond) return;

  // for newton_bond off, also consider atom M as atom 1 in dihedral

  i1 = tag[m];
  n1 = nspecial[m][0];
  s1list = special[m];

  for (i = 0; i < n1; i++) {
    i2 = s1list[i];
    i2local = atom->map(i2);
    s2list = special[i2local];
    n2 = nspecial[i2local][0];

    for (j = 0; j < n2; j++) {
      i3 = s2list[j];
      if (i3 == i1) continue;
      i3local = atom->map(i3);
      s3list = special[i3local];
      n3 = nspecial[i3local][0];

      for (k = 0; k < n3; k++) {
        i4 = s3list[k];
        if (i4 == i1 || i4 == i2 || i4 == i3) continue;

        // dihedral = i1-i2-i3-i4

        for (n = 0; n < ncreate; n++) {
          if (created[n][0] == i1 && created[n][1] == i2) break;
          if (created[n][0] == i2 && created[n][1] == i1) break;
          if (created[n][0] == i2 && created[n][1] == i3) break;
          if (created[n][0] == i3 && created[n][1] == i2) break;
          if (created[n][0] == i3 && created[n][1] == i4) break;
          if (created[n][0] == i4 && created[n][1] == i3) break;
        }
        if (n < ncreate) {

          // check types 
            dcheck = check_dtype(i1,i2,i3,i4,dihedrals); 
            if(!dcheck)
              continue; 
        // dihedrals zero index has number
        // followed by dihedral types
        // loop over dihedrals match in template, add them
        for(int zz = 1; zz <= dihedrals[0]; zz++){
          dtype = dihedrals[zz];
          if (num_dihedral < atom->dihedral_per_atom) {
            dihedral_type[num_dihedral] = dtype;
            dihedral_atom1[num_dihedral] = i1;
            dihedral_atom2[num_dihedral] = i2;
            dihedral_atom3[num_dihedral] = i3;
            dihedral_atom4[num_dihedral] = i4;
            num_dihedral++;
            ndihedrals++;
          } else overflow = 1;
        }

        }
      }
    }
  }

  atom->num_dihedral[m] = num_dihedral;
}

/* ----------------------------------------------------------------------
   create any impropers owned by atom M induced by newly created bonds
   walk special list to find all possible impropers to create
   only add an improper if a new bond is one of its 3 bonds (I-J,I-K,I-L)
   for newton_bond on, atom M is central atom
   for newton_bond off, atom M is any of 4 atoms in improper
------------------------------------------------------------------------- */

void Topo::create_impropers(int m)
{
  int i,j,k,n,i1local,n1,n2;
  tagint i1,i2,i3,i4;
  tagint *s1list,*s2list;

  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  int num_improper = atom->num_improper[m];
  int *improper_type = atom->improper_type[m];
  tagint *improper_atom1 = atom->improper_atom1[m];
  tagint *improper_atom2 = atom->improper_atom2[m];
  tagint *improper_atom3 = atom->improper_atom3[m];
  tagint *improper_atom4 = atom->improper_atom4[m];

  // atom M is central atom in improper
  // triple loop over 1-2 neighs
  // avoid double counting by 2nd loop as j = i+1,N not j = 1,N
  // consider all impropers, only add if:
  //   a new bond is in the improper and atom types match

  i1 = tag[m];
  n1 = nspecial[m][0];
  s1list = special[m];

  for (i = 0; i < n1; i++) {
    i2 = s1list[i];
    for (j = i+1; j < n1; j++) {
      i3 = s1list[j];
      for (k = j+1; k < n1; k++) {
        i4 = s1list[k];

        // improper = i1-i2-i3-i4

        for (n = 0; n < ncreate; n++) {
          if (created[n][0] == i1 && created[n][1] == i2) break;
          if (created[n][0] == i2 && created[n][1] == i1) break;
          if (created[n][0] == i1 && created[n][1] == i3) break;
          if (created[n][0] == i3 && created[n][1] == i1) break;
          if (created[n][0] == i1 && created[n][1] == i4) break;
          if (created[n][0] == i4 && created[n][1] == i1) break;
        }
        if (n == ncreate) continue;

        itype = check_itype(i1,i2,i3,i4); 
        if(!itype) continue; 

        if (num_improper < atom->improper_per_atom) {
          improper_type[num_improper] = itype;
          improper_atom1[num_improper] = i1;
          improper_atom2[num_improper] = i2;
          improper_atom3[num_improper] = i3;
          improper_atom4[num_improper] = i4;
          num_improper++;
          nimpropers++;
        } else overflow = 1;
      }
    }
  }

  atom->num_improper[m] = num_improper;
  if (force->newton_bond) return;

  // for newton_bond off, also consider atom M as atom 2 in improper

  i2 = tag[m];
  n2 = nspecial[m][0];
  s2list = special[m];

  for (i = 0; i < n2; i++) {
    i1 = s2list[i];
    i1local = atom->map(i1);
    s1list = special[i1local];
    n1 = nspecial[i1local][0];

    for (j = 0; j < n1; j++) {
      i3 = s1list[j];
      if (i3 == i1 || i3 == i2) continue;

      for (k = j+1; k < n1; k++) {
        i4 = s1list[k];
        if (i4 == i1 || i4 == i2) continue;

        // improper = i1-i2-i3-i4

        for (n = 0; n < ncreate; n++) {
          if (created[n][0] == i1 && created[n][1] == i2) break;
          if (created[n][0] == i2 && created[n][1] == i1) break;
          if (created[n][0] == i1 && created[n][1] == i3) break;
          if (created[n][0] == i3 && created[n][1] == i1) break;
          if (created[n][0] == i1 && created[n][1] == i4) break;
          if (created[n][0] == i4 && created[n][1] == i1) break;
        }
        if (n < ncreate) {

          itype = check_itype(i3,i2,i1,i4); 
          if(!itype) continue; 

          if (num_improper < atom->improper_per_atom) {
            improper_type[num_improper] = dtype;
            improper_atom1[num_improper] = i1;
            improper_atom2[num_improper] = i2;
            improper_atom3[num_improper] = i3;
            improper_atom4[num_improper] = i4;
            num_improper++;
            nimpropers++;
          } else overflow = 1;
        }
      }
    }
  }
  atom->num_improper[m] = num_improper;
}

/* ----------------------------------------------------------------------
   remove all ID duplicates in copy from Nstart:Nstop-1
   compare to all previous values in copy
   return N decremented by any discarded duplicates
------------------------------------------------------------------------- */

int Topo::dedup(int nstart, int nstop, tagint *copy)
{
  int i;

  int m = nstart;
  while (m < nstop) {
    for (i = 0; i < m; i++)
      if (copy[i] == copy[m]) {
        copy[m] = copy[nstop-1];
        nstop--;
        break;
      }
    if (i == m) m++;
  }

  return nstop;
}

/* ---------------------------------------------------------------------- 
 compare the atom types of each bond against the template
 ---------------------------------------------------------------------- */

int Topo::check_btype(int i1, int i2)
{
  // new code
  int* type = atom->type;
  int  nrbonds = onemol->nrbonds;
  int** bond_ratom = onemol->bond_ratom;
  int i1type = type[i1];
  int i2type = type[i2];
  int i1t = i1type - 1;
  int i2t = i2type - 1;
  
  for (int m = 0; m < nrbonds; m++){
    if (bond_ratom[m][1] == i1type) 
      if (bond_ratom[m][2] == i2type) 
        return bond_ratom[m][0];
    if (bond_ratom[m][2] == i1type) 
      if (bond_ratom[m][1] == i2type) 
        return bond_ratom[m][0];
  }

  // no match
  return 0;
}

/* ---------------------------------------------------------------------- 
compare the atom types of each angle against the template
for any newton setting, this is called for each new angle of atoms i1 i2 i3 
 ---------------------------------------------------------------------- */

int Topo::check_atype(int i1, int i2, int i3)
{

  int* type = atom->type;
  int  nrangles = onemol->nrangles;
  int** angle_ratom = onemol->angle_ratom;
  int i1type = type[atom->map(i1)];
  int i2type = type[atom->map(i2)];
  int i3type = type[atom->map(i3)];

  // look up angles by atom type using molecule template
  // molecule template contains atom types
  // molecule template is zero indexed

  int i1t = i1type - 1; 
  int i2t = i2type - 1; 
  int i3t = i3type - 1;
  
  for (int m = 0; m < nrangles; m++){
    if (angle_ratom[m][2] == i2type){
      // middle atom matched, try ends 
      if (angle_ratom[m][1] == i1type) 
        if (angle_ratom[m][3] == i3type) 
        return angle_ratom[m][0];
      if (angle_ratom[m][3] == i1type) 
        if (angle_ratom[m][1] == i3type) 
        return angle_ratom[m][0];
    } 
  }

  // no match
  return 0;
}

/* ---------------------------------------------------------------------- 
compare the atom types of each dihedral against the template
allow multiple dihedrals to be applied to same four atoms
search template, apply a dihedral each time a match is found
do not apply multiple dihedrals of the same type 
 ---------------------------------------------------------------------- */

int Topo::check_dtype(int i1, int i2, int i3, int i4, int* dihedrals)
{

  int* type = atom->type;
  int i1type = type[atom->map(i1)];
  int i2type = type[atom->map(i2)];
  int i3type = type[atom->map(i3)];
  int i4type = type[atom->map(i4)];
  int i1t = i1type - 1;
  int i2t = i2type - 1;
  int i3t = i3type - 1;
  int i4t = i4type - 1;
  int nrdihedrals = onemol->nrdihedrals;
  int** dihedral_ratom = onemol->dihedral_ratom;
  int i = 1;
  int nfound = 0;

  for (int m = 0; m < nrdihedrals; m++){
    if (dihedral_ratom[m][1] == i1type)
      if (dihedral_ratom[m][2] == i2type)
        if (dihedral_ratom[m][3] == i3type)
          if (dihedral_ratom[m][4] == i4type){
            dihedrals[i++] = dihedral_ratom[m][0];
            nfound++;
          }
    if (dihedral_ratom[m][4] == i1type)
      if (dihedral_ratom[m][3] == i2type)
        if (dihedral_ratom[m][2] == i3type)
          if (dihedral_ratom[m][1] == i4type){
            dihedrals[i++] = dihedral_ratom[m][0];
            nfound++;
          }
  } 

// ensure all new dihedrals have different types
// eliminate duplicate types

int Nc = nfound + 1;
int j;

for(i = 1; i < Nc; i++)
  for(j = i+1; j < Nc; j++)
    if(dihedrals[i] == dihedrals[j]){
      dihedrals[j] = dihedrals[Nc-1];
      Nc--;
      j--;
    } 

// store number of dihedral types
dihedrals[0] = Nc - 1; 

return nfound;
}

/* ---------------------------------------------------------------------- 
compare the atom types of each improper against the template
 ---------------------------------------------------------------------- */

int Topo::check_itype(int i1, int i2, int i3, int i4)
{

  int* type = atom->type;
  int i1type = type[atom->map(i1)];
  int i2type = type[atom->map(i2)];
  int i3type = type[atom->map(i3)];
  int i4type = type[atom->map(i4)];

  // check for matching types in the improper section of molecule template 
  // template is zero indexed

  int i1t = i1type - 1; 
  int i2t = i2type - 1; 
  int i3t = i3type - 1; 
  int i4t = i4type - 1; 
  int nrimpropers = onemol->nrimpropers;
  int** improper_ratom = onemol->improper_ratom;
  int i = 1; 
  int nfound = 0; 

  for (int m = 0; m < nrimpropers; m++){
    if (improper_ratom[m][1] == i1type)
      if (improper_ratom[m][2] == i2type)
        if (improper_ratom[m][3] == i3type)
          if (improper_ratom[m][4] == i4type)
              return improper_ratom[m][0];
    if (improper_ratom[m][4] == i1type)
      if (improper_ratom[m][3] == i2type)
        if (improper_ratom[m][2] == i3type)
          if (improper_ratom[m][1] == i4type)
              return improper_ratom[m][0];
  } 

  // no match
  return 0;
}

/* ----------------------------------------------------------------------
   insure all atoms 2 hops away from owned atoms are in ghost list
   this allows dihedral 1-2-3-4 to be properly created
     and special list of 1 to be properly updated
   if I own atom 1, but not 2,3,4, and bond 3-4 is added
     then 2,3 will be ghosts and 3 will store 4 as its finalpartner
------------------------------------------------------------------------- */

void Topo::check_ghosts()
{
  int i,j,n;
  tagint *slist;

  int **nspecial = atom->nspecial;
  int **special = atom->special;
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
    error->all(FLERR,"Topo needs ghost atoms from further away");
  lastcheck = update->ntimestep;
}

/* ----------------------------------------------------------------------
remove influenced atoms if local and rxn type -1 
------------------------------------------------------------------------- */
//void Topo::delete_atom()
//{
//}

/* ----------------------------------------------------------------------
   break any angles owned by atom M that include atom IDs 1 and 2
   angle is broken if ID1-ID2 is one of 2 bonds in angle (I-J,J-K)
------------------------------------------------------------------------- */

void Topo::break_angles(int m, tagint id1, tagint id2)
{

  int j,found;

  int num_angle = atom->num_angle[m];
  int *angle_type = atom->angle_type[m];
  tagint *angle_atom1 = atom->angle_atom1[m];
  tagint *angle_atom2 = atom->angle_atom2[m];
  tagint *angle_atom3 = atom->angle_atom3[m];

  int i = 0;
  while (i < num_angle) {
    found = 0;
    if (angle_atom1[i] == id1 && angle_atom2[i] == id2) found = 1;
    else if (angle_atom2[i] == id1 && angle_atom3[i] == id2) found = 1;
    else if (angle_atom1[i] == id2 && angle_atom2[i] == id1) found = 1;
    else if (angle_atom2[i] == id2 && angle_atom3[i] == id1) found = 1;
    if (!found) i++;
    else {
      for (j = i; j < num_angle-1; j++) {
        angle_type[j] = angle_type[j+1];
        angle_atom1[j] = angle_atom1[j+1];
        angle_atom2[j] = angle_atom2[j+1];
        angle_atom3[j] = angle_atom3[j+1];
      }
      num_angle--;
      nangles++;
    }
  }

  atom->num_angle[m] = num_angle;
}

/* ----------------------------------------------------------------------
   break any dihedrals owned by atom M that include atom IDs 1 and 2
   dihedral is broken if ID1-ID2 is one of 3 bonds in dihedral (I-J,J-K.K-L)
------------------------------------------------------------------------- */

void Topo::break_dihedrals(int m, tagint id1, tagint id2)
{
  int j,found;

  int num_dihedral = atom->num_dihedral[m];
  int *dihedral_type = atom->dihedral_type[m];
  tagint *dihedral_atom1 = atom->dihedral_atom1[m];
  tagint *dihedral_atom2 = atom->dihedral_atom2[m];
  tagint *dihedral_atom3 = atom->dihedral_atom3[m];
  tagint *dihedral_atom4 = atom->dihedral_atom4[m];

  int i = 0;
  while (i < num_dihedral) {
    found = 0;
    if (dihedral_atom1[i] == id1 && dihedral_atom2[i] == id2) found = 1;
    else if (dihedral_atom2[i] == id1 && dihedral_atom3[i] == id2) found = 1;
    else if (dihedral_atom3[i] == id1 && dihedral_atom4[i] == id2) found = 1;
    else if (dihedral_atom1[i] == id2 && dihedral_atom2[i] == id1) found = 1;
    else if (dihedral_atom2[i] == id2 && dihedral_atom3[i] == id1) found = 1;
    else if (dihedral_atom3[i] == id2 && dihedral_atom4[i] == id1) found = 1;
    if (!found) i++;
    else {
      for (j = i; j < num_dihedral-1; j++) {
        dihedral_type[j] = dihedral_type[j+1];
        dihedral_atom1[j] = dihedral_atom1[j+1];
        dihedral_atom2[j] = dihedral_atom2[j+1];
        dihedral_atom3[j] = dihedral_atom3[j+1];
        dihedral_atom4[j] = dihedral_atom4[j+1];
      }
      num_dihedral--;
      ndihedrals++;
    }
  }
  atom->num_dihedral[m] = num_dihedral;
}

/* ----------------------------------------------------------------------
   break any impropers owned by atom M that include atom IDs 1 and 2
   improper is broken if ID1-ID2 is one of 3 bonds in improper (I-J,I-K,I-L)
------------------------------------------------------------------------- */

void Topo::break_impropers(int m, tagint id1, tagint id2)
{
  int j,found;

  int num_improper = atom->num_improper[m];
  int *improper_type = atom->improper_type[m];
  tagint *improper_atom1 = atom->improper_atom1[m];
  tagint *improper_atom2 = atom->improper_atom2[m];
  tagint *improper_atom3 = atom->improper_atom3[m];
  tagint *improper_atom4 = atom->improper_atom4[m];

  int i = 0;
  while (i < num_improper) {
    found = 0;
    if (improper_atom1[i] == id1 && improper_atom2[i] == id2) found = 1;
    else if (improper_atom1[i] == id1 && improper_atom3[i] == id2) found = 1;
    else if (improper_atom1[i] == id1 && improper_atom4[i] == id2) found = 1;
    else if (improper_atom1[i] == id2 && improper_atom2[i] == id1) found = 1;
    else if (improper_atom1[i] == id2 && improper_atom3[i] == id1) found = 1;
    else if (improper_atom1[i] == id2 && improper_atom4[i] == id1) found = 1;
    if (!found) i++;
    else {
      for (j = i; j < num_improper-1; j++) {
        improper_type[j] = improper_type[j+1];
        improper_atom1[j] = improper_atom1[j+1];
        improper_atom2[j] = improper_atom2[j+1];
        improper_atom3[j] = improper_atom3[j+1];
        improper_atom4[j] = improper_atom4[j+1];
      }
      num_improper--;
      nimpropers++;
    }
  }

  atom->num_improper[m] = num_improper;
}


/* ----------------------------------------------------------------------
   comm to update special lists 
------------------------------------------------------------------------- */

void Topo::comm_special()
{

  // setup for comm of special lists
  // need only 1-2 neighs for ghosts
  // zero array for special comm 
  // use double arrays for now
  // pack ints into double later?
 
  int i,j;
  int nlocal = atom->nlocal;
  int nall = nlocal+atom->nghost;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  tagint *slist = special[i];
 
  if(atom->nmax > maxnmax){
    memory->grow(test,atom->nmax,2+atom->maxspecial,"topo:test");
    maxnmax = atom->nmax;
  }

  for (i = 0; i < nall; i++) 
    for (j = 0; j < 2+atom->maxspecial; j++) 
      test[i][j] = 0.0;

// pack array for special comm
// only pack 1-2 neighs
// use forward comm of array
// pack special, nspecial into one array so there is only one comm

  int ns;

  for (i = 0; i < nlocal; i++){
    ns = nspecial[i][0];
    test[i][0] = (double)ns;
    slist = special[i];
    for (j = 1; j <= ns; j++)
      test[i][j] = (double)slist[j-1];
  }

  comm->forward_comm_array(2+atom->maxspecial,test);

  // unpack special
  for (i = nlocal; i < nall; i++){
    nspecial[i][0] = (int) ubuf(test[i][0]).i;
    nspecial[i][1] = (int) ubuf(test[i][0]).i;
    nspecial[i][2] = (int) ubuf(test[i][0]).i;
    ns = nspecial[i][0];
    for (j = 1; j <= ns; j++){
      special[i][j-1] = (int) ubuf(test[i][j]).i;
    }
  }
}

