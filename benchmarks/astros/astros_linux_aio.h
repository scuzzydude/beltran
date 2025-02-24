#ifndef __ASTROS_LINUX_AIO
#define __ASTROS_LINUX_AIO




/* Linux Asynchronous IO */
int astros_aio_prep_ccb(void *pvEngine, ccb *pCCB, target *pTarget, void * pvfnCallback);

/* Uring */

int astros_iouring_init(void *pvEngine);
int astros_iouring_free(void *pvEngine);
int astros_iouring_setup_ccb(void *pvEngine, ccb *pCCB);


int astros_iouring_reset(void *pvEngine);
int astros_iouring_register(void *pvEngine);
int astros_iouring_queue_pending(void *pvEngine);
int astros_iouring_complete(void *pvEngine, bool bDrain);


int astros_libaio_init(void *pvEngine);
int astros_libaio_free(void *pvEngine);
int astros_libaio_setup_ccb(void *pvEngine, ccb *pCCB);


int astros_libaio_reset(void *pvEngine);
int astros_libaio_register(void *pvEngine);
int astros_libaio_queue_pending(void *pvEngine);
int astros_libaio_complete(void *pvEngine, bool bDrain);

int astros_aiowin_init(void *pvEngine);
int astros_aiowin_free(void *pvEngine);
int astros_aiowin_setup_ccb(void *pvEngine, ccb *pCCB);
int astros_aiowin_prep_ccb(void *pvEngine, ccb *pCCB, target *pTarget, void * pvfnCallback);
int astros_aiowin_register(void *pvEngine);
int astros_aiowin_queue_pending(void *pvEngine);
int astros_aiowin_complete(void *pvEngine, bool bDrain);
int astros_open_target(astros_lineup * pLineup, char *path, int targetidx, int operation);



#endif
