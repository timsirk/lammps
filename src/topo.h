/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
#ifndef LMP_TOPO_H
#define LMP_TOPO_H

#include "pointers.h"

namespace LAMMPS_NS {

class Topo : protected Pointers {
 public:
 Topo(class LAMMPS *);
 ~Topo();
  int printBonds(bigint, int**);
  int change_bonds(int, int*, Molecule*);
  class Molecule *onemol;

  // for topo manipulation
  void check_ghosts();
  void update_topology();
  void rebuild_special(int);
  void create_angles(int);
  void create_dihedrals(int);
  void create_impropers(int);
  void break_angles(int, tagint, tagint);
  void break_dihedrals(int, tagint, tagint);
  void break_impropers(int, tagint, tagint);
  int dedup(int, int, tagint*);

  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *);
  void comm_special();
private:

  // working variables
  int nangles, ndihedrals, nimpropers;
  int btype, atype, dtype, itype;
  int angleflag, dihedralflag, improperflag;
  int flag;
  int me; 
  int mode;
  int overflow;
  int calledcreateangles;
  int commflag;

  int *finalpartner;
  int maxnmax;
  // for changing atom types 
  int *influencedlist;
  int maxinfluenced;
  int ncreate,ncreatelocal,maxcreate;
  int nbreak,nbreaklocal,maxbreak;
  int **altered; 
  tagint *copy;
  tagint lastcheck;
  int *dlist;
  int gbit, igbit;

  double **temp;
  // union data struct for packing 32-bit and 64-bit ints into double bufs
  // see atom_vec.h for documentation
  union ubuf {
    double d;
    int64_t i;
    ubuf(double arg) : d(arg) {}
    ubuf(int64_t arg) : i(arg) {}
    ubuf(int arg) : i(arg) {}
  };
  
  std::map<tagint,int> *hash;
  void delete_atom();
  void delete_bond();
  static void bondring(int nbuf, char *cbuf);
  static Topo *cptr;

  // for type matching
  int check_btype(int, int);
  int check_atype(int, int, int);
  int check_dtype(int, int, int, int, int*);
  int check_itype(int, int, int, int);
};
}

#endif
