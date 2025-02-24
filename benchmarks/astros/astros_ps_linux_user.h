#ifndef _ASTROS_PS_LINUX_USER_H
#define _ASTROS_PS_LINUX_USER_H

#include <stdio.h>
#include <stdint.h>
#include <errno.h>
#include <assert.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdarg.h>
#include <string.h>
#include <ctype.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <dirent.h>



#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <sys/resource.h>
#include <sys/ioctl.h>
#include <sys/utsname.h>


#ifdef ASTROS_CYGWIN
#else
#include <libaio.h>
#include "liburing.h"
#endif

/* Platform Specific stuff */


#define ASTROS_ASSERT(__assert) assert(__assert)

#define ASTROS_DBG_PRINT(__verbose, __format, ...) do { if( __verbose >= ASTROS_DBGLVL_COMPILE) astros_dbg_printf(__verbose, __format, __VA_ARGS__); } while (0)\



#if 1

#define ASTROS_GET_CPU(_pCPU) *(_pCPU) = sched_getcpu()

#else
#define ASTROS_GET_CPU(_pCPU) getcpu(_pCPU, NULL)
#endif



static inline void astros_dbg_printf(int verbose, const char *fmt, ...)
{

    va_list args;
   
    if(verbose >=  ASTROS_DBGLVL_COMPILE )
    {   
            
        va_start(args, fmt);
        vprintf(fmt, args);
        va_end(args);
    }

}




#if 0//def ASTROS_CYGWIN
extern UINT64 g_win_nano_multiplier;

static inline UINT64 astros_get_hr_res_ns(int clockid)
{
    return g_win_nano_multiplier;

}

static inline UINT64 astros_hrclk_get(void)
{

    int verbose = ASTROS_DBGLVL_NONE;

	LARGE_INTEGER ticks;
    UINT64 ns;
	
	QueryPerformanceCounter(&ticks);

	ns = (ticks.QuadPart * g_win_nano_multiplier);

	ASTROS_DBG_PRINT(verbose, "NS: %lld\n", ns);	
	
    return ns;


}
#else
static inline UINT64 astros_get_hr_res_ns(int clockid)
{
    UINT64 res_ns = 0;
    struct timespec res;

    if(0 == clock_getres(clockid, &res))
    {   
        res_ns = (res.tv_sec * 1000000000L) + res.tv_nsec;
    }

    return res_ns;

}


static inline UINT64 astros_hrclk_get(void)
{
    UINT64 ns = 0;
    struct timespec res;

	
		
    if(0 == clock_gettime(CLOCK_MONOTONIC, &res))
    {   
        ns = (res.tv_sec * 1000000000L) + res.tv_nsec;
    }

    return ns;


}

#endif
#define ASTROS_PS_GET_HRCLK_RES_NS(_id) astros_get_hr_res_ns(_id)



#define ASTROS_BATTER_JERSEY pthread_t

static inline int astros_batter_set_cpu(ASTROS_BATTER_JERSEY jersey, int cpu)
{
    int err = 0;
	int verbose = ASTROS_DBGLVL_NONE;
	int cssize = sizeof(cpu_set_t);
	cpu_set_t cpuset;
    
    CPU_ZERO(&cpuset);

    CPU_SET(cpu, &cpuset);
    err = pthread_setaffinity_np(jersey, sizeof(cpu_set_t), &cpuset);

	ASTROS_DBG_PRINT(verbose, "astros_batter_set_cpu(%px:%d) err = %d cssize = %d\n", jersey, cpu, err, cssize);
    return err;
}



int astros_cygwin_rotation_create(void * pVfn, void * pvBatter, int cpu);


#define ASTROS_PS_HRCLK_GET() astros_hrclk_get()


#define ASTROS_PS_GET_NCPUS()  get_nprocs()

#define ASTROS_THREAD_FN_RET void * 


//#ifdef ASTROS_CYGWIN
//#define ASTROS_BATTER_ROTATION_CREATE(__pBatter, __rotation_func, __cpu)  astros_cygwin_rotation_create(__rotation_func, __pBatter, __cpu)
//#else
#define ASTROS_BATTER_ROTATION_CREATE(__pBatter, __rotation_func, __cpu) pthread_create(&__pBatter->id, NULL, __rotation_func, __pBatter)
//#endif

#define ASTROS_BATTER_SET_CPU(__jersey, _cpu) astros_batter_set_cpu(__jersey, _cpu) 




#define ASTROS_BATTER_SLEEPUS(_us) usleep(_us)
#define ASTROS_BATTER_SET_PRIORITY(__pBatter, __pri) astros_batter_set_priority(__pBatter, __pri)



#define ASTROS_BATTER_GET_POLICY_AND_PRIORITY(__pBatter, __pol, __pri) astros_batter_get_policy_and_priority(__pBatter, __pol, __pri)


static inline void * astros_aligned_alloc(size_t align, size_t size)
{
    void *ptr = aligned_alloc(align, size);

    if(ptr)
    {
        memset(ptr, 0, size);
    }

    return ptr;
}

#define ASTROS_ALLOC(__align, __numbytes) astros_aligned_alloc(__align, __numbytes)
#define ASTROS_ALLOC_DATABUFFER(__ppPtr, __align, __numbytes) posix_memalign(__ppPtr, __align, __numbytes)
#define ASTROS_FREE(_ptr) free(_ptr)
#define ASTROS_FREE_DATABUFFER(__ptr, __numbytes) free(__ptr)


#define ASTROS_ATOMIC  int
#define ASTROS_ATOMIC64  long
#define ASTROS_INC_ATOMIC(__ptr) __sync_fetch_and_add(__ptr, 1)
#define ASTROS_DEC_ATOMIC(__ptr) __sync_fetch_and_sub(__ptr, 1)
#define ASTROS_GET_ATOMIC(__atomic) (__atomic)
#define ASTROS_ADD_ATOMIC(__atomic, __val) __sync_fetch_and_add(&__atomic, __val)
#define ASTROS_SET_ATOMIC(__atomic, __val) __sync_lock_test_and_set(&__atomic, __val)

#define ASTROS_SPINLOCK pthread_spinlock_t
#define ASTROS_SPINLOCK_INIT(__splck) pthread_spin_init(&__splck, PTHREAD_PROCESS_PRIVATE)
#define ASTROS_SPINLOCK_DESTROY(__splck) pthread_spin_destroy(&__splck)
#define ASTROS_SPINLOCK_TRYLOCK(__splck) pthread_spin_trylock(&__splck)
#define ASTROS_SPINLOCK_LOCK(__splck) pthread_spin_lock(&__splck)
#define ASTROS_SPINLOCK_UNLOCK(__splck) pthread_spin_unlock(&__splck)

#define ASTROS_SPINLOCK_IRQSAVE(__splck, __ulflags) pthread_spin_lock(&__splck)
#define ASTROS_SPINUNLOCK_IRQSAVE(__splck, __ulflags) pthread_spin_lock(&__splck)


#define ASTROS_IO_URING_STRUCT struct io_uring
#define ASTROS_FILE_PTR        FILE *

#define ASTROS_WAIT_SCORER_READY(__scard) astros_wait_scorer_ready(__scard)
#define ASTROS_SCORE_CSVLOG(__ic, __plineup, __scard) astros_score_csvlog(__ic, __plineup, __scard)
#define ASTROS_SCORER_POST_FINAL(__plineup, __scard) if(__plineup->field == ASTROS_FIELD_USER) astros_scorer_post_final(__plineup, __scard)
#define ASTROS_SCORE_INNING(__ic, __scard) astros_score_inning(__ic, __scard)

#define ASTROS_UMPIRE_PREGAME(__lu) astros_umpire_pregame(__lu)

#define ASTROS_FREE_SCORECARD(_sc) if(_sc) ASTROS_FREE(_sc)


#define ASTROS_GET_RAND32() rand()

#define ASTROS_GETHOSTNAME(__buf, __size) gethostname(__buf, __size)
#define ASTROS_SYNC_READ(_fd, _buf, _io_size, _off) pread(_fd, _buf, _io_size, _off)
#define ASTROS_SYNC_WRITE(_fd, _buf, _io_size, _off) pwrite(_fd, _buf, _io_size, _off)






static inline uint64_t get_cycles()
{
  uint64_t t;
  __asm volatile ("rdtsc" : "=A"(t));
  return t;
}


/* special macros use without a trailing (;) - if calling code or just null function call and include the (;) in macro definition */
/* if we want to compile out the floating point block, define as if(0) */
#define ASTROS_FP_BEGIN()

#define ASTROS_FP_END()


static inline void astros_dump_memory(char *szSearchString, char *szMarker)
{

	int verbose = ASTROS_DBGLVL_INFO;

	pid_t pid;
	FILE* fd;
    // linux file contains this-process info
	char fn[128]; 
	int size = 1024;
	char *ptr;
	int count = 0;

	pid = getpid();

	sprintf(fn, "/proc/%d/status", pid);

	
	//stat(fn, &st);
	//size = st.st_size;

	ASTROS_DBG_PRINT(verbose, "---------(%s) astros_dump_memory(%d:%d) %s ---------------------------------------\n", szMarker, pid, size, fn);


	if(size)
	{
		ptr = (char *)ASTROS_ALLOC(64, size);

		ASTROS_ASSERT(ptr);
	

		fd = fopen(fn, "r");

		if(fd)
		{
			while(fgets( ptr, size, fd))
			{
				if(szSearchString)
				{
					if(strstr(ptr,szSearchString))
					{
						printf("+[%d] STATUS:%s\n", count, ptr);
						count++;
						break;					
					}
				}
				else
				{
					printf("+[%d] STATUS:%s\n", count, ptr);
					count++;
				}
						
			}
			
			fclose(fd);

		}

		ASTROS_FREE(ptr);

	}

}

#define ASTROS_DUMP_MEMORY(_search_string, _marker) astros_dump_memory(_search_string, _marker)

#ifdef ASTROS_CYGWIN
#define ASTROS_FD HANDLE
#else
#define ASTROS_FD int 
#endif


#ifdef ASTROS_LIBAIO

#define ASTROS_LIBAIO_CTX io_context_t
#else
#define ASTROS_LIBAIO_CTX int
#endif
#define ZOMBIE_QCMD_FN void *
















#ifdef ASTROS_LIBURING

typedef struct
{
    ASTROS_IO_URING_STRUCT ring;
    struct iovec *pIov;
    int    cur_iov_idx;



	
} iouring_engine_spec;

#endif

#ifdef ASTROS_LIBAIO

typedef struct
{

	ASTROS_LIBAIO_CTX ioctx;
	struct iocb **iocbA;
	int    idx;
	struct io_event *pEvent;

} libaio_engine_spec;
#endif


#endif
