#ifndef _ASTROS_PS_WINDOWS_H
#define _ASTROS_PS_WINDOWS_H


#include <Windows.h>


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
//#include <unistd.h>
//#include <pthread.h>
//#include <sched.h>
//#include <dirent.h>
//#include <libaio.h>


#include <crtdbg.h>

#include <sys/stat.h>
//#include <sys/sysinfo.h>
//#include <sys/resource.h>
//#include <sys/ioctl.h>
//#include <sys/utsname.h>


//#include "liburing.h"


/* Platform Specific stuff */


#define ASTROS_ASSERT(__cond) _ASSERT(__cond)

#define ASTROS_DBG_PRINT(__verbose, __format, ...) do { if( __verbose >= ASTROS_DBGLVL_COMPILE) astros_dbg_printf(__verbose, __format, __VA_ARGS__); } while (0)\



#if 1

#define ASTROS_GET_CPU(_pCPU) *(_pCPU) = GetCurrentProcessorNumber()

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



#define ASTROS_PS_GET_HRCLK_RES_NS(_id) astros_get_hr_res_ns(_id)




extern LARGE_INTEGER g_win_ticksPerSec;
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

#define ASTROS_BATTER_JERSEY HANDLE

static inline int astros_batter_set_cpu(ASTROS_BATTER_JERSEY jersey, int cpu)
{
#if 0
    cpu_set_t cpuset;
    int err = 0;
    
    CPU_ZERO(&cpuset);

    CPU_SET(cpu, &cpuset);

    if(pthread_setaffinity_np(jersey, sizeof(cpu_set_t), &cpuset))
    {
       err = -1;
    }

    return err;

#else
	UINT64 one = 1;
	UINT64 mask = one << (UINT64)cpu;

	if(0 == SetThreadAffinityMask(jersey, &mask))
	{
		ASTROS_ASSERT(0);
		return 1;
	}


#endif
	return 0;



}





#define ASTROS_PS_HRCLK_GET() astros_hrclk_get()

static inline int astros_win_get_numcpus(void)
{
	int numCPU;
	SYSTEM_INFO sysinfo;
	GetSystemInfo(&sysinfo);
	numCPU = sysinfo.dwNumberOfProcessors;

	return numCPU;

}
#define ASTROS_PS_GET_NCPUS()  astros_win_get_numcpus()

#define ASTROS_THREAD_FN_RET void * 

int astros_win_rotation_create(void * pVfn, void * pvBatter);
int astros_win_init(void);



#define ASTROS_BATTER_ROTATION_CREATE(__pBatter, __rotation_func, __cpu)  astros_win_rotation_create(__rotation_func, __pBatter)

#define ASTROS_BATTER_SLEEPUS(_us) usleep(_us)
#define ASTROS_BATTER_SET_CPU(__jersey, _cpu) astros_batter_set_cpu(__jersey, _cpu) 
#define ASTROS_BATTER_SET_PRIORITY(__pBatter, __pri) astros_batter_set_priority(__pBatter, __pri)



#define ASTROS_BATTER_GET_POLICY_AND_PRIORITY(__pBatter, __pol, __pri) astros_batter_get_policy_and_priority(__pBatter, __pol, __pri)


static inline void * astros_aligned_alloc(size_t align, size_t size)
{
//    void *ptr = aligned_alloc(align, size);

    void *ptr = _aligned_malloc(size, align );

    if(ptr)
    {
        memset(ptr, 0, size);
    }

    return ptr;
}

#define ASTROS_ALLOC(__align, __numbytes) astros_aligned_alloc(__align, __numbytes)
#define ASTROS_ALLOC_DATABUFFER(__ppPtr, __align, __numbytes) __ppPtr = astros_aligned_alloc(__align, __numbytes)
#define ASTROS_FREE(_ptr) _aligned_free(_ptr)
#define ASTROS_FREE_DATABUFFER(__ptr, __numbytes) free(__ptr)


#define ASTROS_ATOMIC  int
#define ASTROS_INC_ATOMIC(__ptr) InterlockedIncrement(__ptr)
#define ASTROS_DEC_ATOMIC(__ptr) InterlockedDecrement(__ptr)
#define ASTROS_GET_ATOMIC(__atomic) (__atomic)
#define ASTROS_ADD_ATOMIC(__atomic, __val) InterlockedIncrement(&__atomic, __val)
#define ASTROS_SET_ATOMIC(__atomic, __val) InterlockedExchange(&__atomic, __val)




#define WIN_MAX_SPIN_CNT (1024 * 8)
#define ASTROS_SPINLOCK CRITICAL_SECTION
#define ASTROS_SPINLOCK_INIT(__splck) InitializeCriticalSectionAndSpinCount(&__splck, WIN_MAX_SPIN_CNT)
#define ASTROS_SPINLOCK_DESTROY(__splck) DeleteCriticalSection(&__splck)
#define ASTROS_SPINLOCK_TRYLOCK(__splck) TryEnterCriticalSection(&__splck)
#define ASTROS_SPINLOCK_LOCK(__splck) EnterCriticalSection(&__splck)
#define ASTROS_SPINLOCK_UNLOCK(__splck) LeaveCriticalSection(&__splck)

#define ASTROS_SPINLOCK_IRQSAVE(__splck, __ulflags) 
#define ASTROS_SPINUNLOCK_IRQSAVE(__splck, __ulflags) 


#define ASTROS_IO_URING_STRUCT struct io_uring
#define ASTROS_FILE_PTR        FILE *

#define ASTROS_WAIT_SCORER_READY(__scard) astros_wait_scorer_ready(__scard)
#define ASTROS_SCORE_CSVLOG(__ic, __plineup, __scard) astros_score_csvlog(__ic, __plineup, __scard)
#define ASTROS_SCORER_POST_FINAL(__plineup, __scard) if(__plineup->field == ASTROS_FIELD_USER) astros_scorer_post_final(__plineup, __scard)
#define ASTROS_SCORE_INNING(__ic, __scard) astros_score_inning(__ic, __scard)

#define ASTROS_UMPIRE_PREGAME(__lu) astros_umpire_pregame(__lu)

#define ASTROS_FREE_SCORECARD(_sc) if(_sc) ASTROS_FREE(_sc)


#define ASTROS_GET_RAND32() rand()

static inline void astros_get_computer_name(void * buf, int xsize)
{
	int mxsize = xsize;
	TCHAR computer_name[MAX_COMPUTERNAME_LENGTH + 1];
	int slen;
	size_t nNumCharConverted;
	
	GetComputerName(computer_name, &slen);
	
	
	wcstombs_s(&nNumCharConverted, buf, 255,
				computer_name, 255);
		
	
	
}


#define ASTROS_GETHOSTNAME(__buf, __size) astros_get_computer_name(__buf, __size)

static inline astros_win_read(HANDLE hFile, LPVOID lpBuffer, DWORD numbytes, int offset)
{
   int br = 0;
   OVERLAPPED ol = {0};

	ol.Offset = offset;

   if(ReadFile(hFile, lpBuffer, numbytes, &br, &ol) )
   {

   		return br;

   }
   

	return 0;
}

static inline astros_win_write(HANDLE hFile, LPVOID lpBuffer, DWORD numbytes, int offset)
{
   int br = 0;
   OVERLAPPED ol = {0};

	ol.Offset = offset;

   if(WriteFile(hFile, lpBuffer, numbytes, &br, &ol) )
   {

   		return br;

   }

	return 0;
}



#define ASTROS_SYNC_READ(_fd, _buf, _io_size, _off) astros_win_read(_fd, _buf, _io_size, _off)
#define ASTROS_SYNC_WRITE(_fd, _buf, _io_size, _off) astros_win_write(_fd, _buf, _io_size, _off)




static inline size_t getline(char **lineptr, size_t *n, FILE *stream) {
    size_t pos;
    int c;

    if (lineptr == NULL || stream == NULL || n == NULL) {
        errno = EINVAL;
        return -1;
    }

    c = getc(stream);
    if (c == EOF) {
        return -1;
    }

    if (*lineptr == NULL) {
        *lineptr = malloc(128);
        if (*lineptr == NULL) {
            return -1;
        }
        *n = 128;
    }

    pos = 0;
    while(c != EOF) {
        if (pos + 1 >= *n) {
            size_t new_size = *n + (*n >> 2);
            if (new_size < 128) {
                new_size = 128;
            }
            char *new_ptr = realloc(*lineptr, new_size);
            if (new_ptr == NULL) {
                return -1;
            }
            *n = new_size;
            *lineptr = new_ptr;
        }

        ((unsigned char *)(*lineptr))[pos ++] = c;
        if (c == '\n') {
            break;
        }
        c = getc(stream);
    }

    (*lineptr)[pos] = '\0';
    return pos;
}


static inline void usleep(__int64 usec) 
{ 
    HANDLE timer; 
    LARGE_INTEGER ft; 

    ft.QuadPart = -(10*usec); // Convert to 100 nanosecond interval, negative value indicates relative time

    timer = CreateWaitableTimer(NULL, TRUE, NULL); 
    SetWaitableTimer(timer, &ft, 0, NULL, NULL, 0); 
    WaitForSingleObject(timer, INFINITE); 
    CloseHandle(timer); 
}


static inline uint64_t get_cycles()
{

#if 0
  uint64_t t;
  __asm volatile ("rdtsc" : "=A"(t));
  return t;
#else

	return 0;
#endif	

}


/* special macros use without a trailing (;) - if calling code or just null function call and include the (;) in macro definition */
/* if we want to compile out the floating point block, define as if(0) */
#define ASTROS_FP_BEGIN()

#define ASTROS_FP_END()


static inline void astros_dump_memory(char *szSearchString, char *szMarker)
{
#if 0
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
		ptr = ASTROS_ALLOC(64, size);

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

#endif
}

#define ASTROS_DUMP_MEMORY(_search_string, _marker) astros_dump_memory(_search_string, _marker)

#define ASTROS_FD HANDLE


#ifdef ASTROS_LIBAIO

#define ASTROS_LIBAIO_CTX io_context_t
#else
#define ASTROS_LIBAIO_CTX int
#endif


#define ZOMBIE_QCMD_FN void *



typedef struct
{

	UINT32 dummy;

	
} winaio_engine_spec;





#endif
