#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _Im_reg(void);
extern void _Kv3_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," Im.mod");
    fprintf(stderr," Kv3.mod");
    fprintf(stderr, "\n");
  }
  _Im_reg();
  _Kv3_reg();
}
