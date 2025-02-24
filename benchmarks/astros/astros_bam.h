#ifndef __ASTROS_BAM_H
#define __ASTROS_BAM_H

/* Used by both C and C++ so keep it simple, no references to external structures or classes */

#define BAM_MAX_DEVICE_PATH 64
#define BAM_MAX_CONTROLLERS 16


#define BAM_ASTROS_DBGLVL_NONE    0
#define BAM_ASTROS_DBGLVL_IOPATH  1
#define BAM_ASTROS_DBGLVL_ERROR   2
#define BAM_ASTROS_DBGLVL_INFO    3
#define BAM_ASTROS_DBGLVL_DETAIL  4
#define BAM_ASTROS_DBGLVL_VERBOSE 5

#define BAM_ASTROS_DBGLVL_COMPILE BAM_ASTROS_DBGLVL_ERROR

#endif

