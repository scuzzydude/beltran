#ifndef __EMU_STATS_H
#define __EMU_STATS_H

#define EMU_STATS_LVL_NONE       0
#define EMU_STATS_LVL_BASIC      1
#define EMU_STATS_LVL_ADVANCED   2
#define EMU_STATS_LVL_HISTORGRAM 3
#define EMU_STATS_LVL_VENDOR     4

#define EMU_STATS_LEVEL EMU_STATS_LVL_NONE


//Macro Magic Trickery, add standard uint64_t stats in this list 
#define EMU_BASIC_STAT_NAME \
	__Bx(BasicStatRequest) \
	__Bx(BasicStatResponse) \
	__Bx(BasicStatCullStall) \

#define __Bx(x) x, 
enum EmuBasicStatEnum { EMU_BASIC_STAT_NAME BasicStatMaxEnum};

#undef __Bx
#define __Bx(x) #x,
static const char * const EmuBasicStatString[] = { EMU_BASIC_STAT_NAME };
//9 is index of the first unqiue part of the stat i.e. BasicStatX
#define EmuBasicStatStr(x) &EmuBasicStatString[x][9] 






typedef struct
{
	uint64_t	 basic[BasicStatMaxEnum];



} emu_stats;



#define EMU_GET_STATS_PTR(_tmgr)  ((emu_stats *)(&(_tmgr->pStats[(blockIdx.x * blockDim.x + threadIdx.x)])))


#if (EMU_STATS_LEVEL >= EMU_STATS_LVL_BASIC)
#define EMU_STATS_BASIC_INC(__tmgr, __counter) EMU_GET_STATS_PTR(__tmgr)->basic[__counter]++
#else
#define EMU_STATS_BASIC_INC(__tmgr, __counter) 
#endif



#endif

