#ifdef ASTROS_SPDK

#include "spdk/stdinc.h"

#include "spdk/nvme.h"
#include "spdk/vmd.h"
#include "spdk/nvme_zns.h"
#include "spdk/env.h"
#include "spdk/string.h"
#include "spdk/log.h"


#include "astros.h"



struct ctrlr_entry {
	struct spdk_nvme_ctrlr		*ctrlr;
	TAILQ_ENTRY(ctrlr_entry)	link;
	char				name[1024];
};

struct ns_entry {
	struct spdk_nvme_ctrlr	*ctrlr;
	struct spdk_nvme_ns	*ns;
	TAILQ_ENTRY(ns_entry)	link;
	struct spdk_nvme_qpair	*qpair;
};

static TAILQ_HEAD(, ctrlr_entry) g_controllers = TAILQ_HEAD_INITIALIZER(g_controllers);
static TAILQ_HEAD(, ns_entry) g_namespaces = TAILQ_HEAD_INITIALIZER(g_namespaces);
static struct spdk_nvme_transport_id g_trid = {};


static void
register_ns(struct spdk_nvme_ctrlr *ctrlr, struct spdk_nvme_ns *ns)
{
	struct ns_entry *entry;

	if (!spdk_nvme_ns_is_active(ns)) {
		return;
	}

	entry = malloc(sizeof(struct ns_entry));
	if (entry == NULL) {
		perror("ns_entry malloc");
		exit(1);
	}

	entry->ctrlr = ctrlr;
	entry->ns = ns;
	TAILQ_INSERT_TAIL(&g_namespaces, entry, link);

	printf("  Namespace ID: %d size: %juGB\n", spdk_nvme_ns_get_id(ns),
	       spdk_nvme_ns_get_size(ns) / 1000000000);
}


static bool g_hex_dump = false;

static int g_shm_id = -1;

static int g_dpdk_mem = 0;

static bool g_dpdk_mem_single_seg = false;

static int g_main_core = 0;

static char g_core_mask[20] = "0x1";

static char g_hostnqn[SPDK_NVMF_NQN_MAX_LEN + 1];

static int g_controllers_found = 0;

static bool g_vmd = false;

static bool g_ocssd_verbose = false;

static struct spdk_nvme_detach_ctx *g_detach_ctx = NULL;
static bool
probe_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
	 struct spdk_nvme_ctrlr_opts *opts)
{
	memcpy(opts->hostnqn, g_hostnqn, sizeof(opts->hostnqn));
	return true;
}

static void
attach_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
	  struct spdk_nvme_ctrlr *ctrlr, const struct spdk_nvme_ctrlr_opts *opts)
{
	int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
	g_controllers_found++;

	ASTROS_DBG_PRINT(verbose, "attach_cb(%d)\n", g_controllers_found);

//	print_controller(ctrlr, trid, opts);
//	spdk_nvme_detach_async(ctrlr, &g_detach_ctx);
}


int astros_spdk_test(void)
{

	int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
	int 			rc;
	struct spdk_env_opts		opts;
	struct spdk_nvme_ctrlr		*ctrlr;


	spdk_env_opts_init(&opts);
	opts.name = "astros";
	opts.shm_id = g_shm_id;
	opts.mem_size = g_dpdk_mem;
	opts.mem_channel = 1;
opts.main_core = g_main_core;
opts.core_mask = g_core_mask;
opts.hugepage_single_segments = g_dpdk_mem_single_seg;

	g_trid.trtype = SPDK_NVME_TRANSPORT_PCIE;
	strncpy(g_trid.trstring, SPDK_NVME_TRANSPORT_NAME_PCIE, SPDK_NVMF_TRSTRING_MAX_LEN);


	strncpy(g_trid.subnqn, SPDK_NVMF_DISCOVERY_NQN, strlen(SPDK_NVMF_DISCOVERY_NQN));
	
		

	ASTROS_DBG_PRINT(verbose, "g_trid.trtype = %d\n", g_trid.trtype);
	ASTROS_DBG_PRINT(verbose, "g_trid.trstring = %s\n", g_trid.trstring);
	ASTROS_DBG_PRINT(verbose, "g_trid.subnqn = %s\n", g_trid.subnqn);
	


if (g_trid.trtype != SPDK_NVME_TRANSPORT_PCIE) 
{
	opts.no_pci = true;
}

if (spdk_env_init(&opts) < 0) 
{
	fprintf(stderr, "Unable to initialize SPDK env\n");
	return 1;
}

if (g_vmd && spdk_vmd_init()) 
{
	fprintf(stderr, "Failed to initialize VMD."
		" Some NVMe devices can be unavailable.\n");
}
else
{
	ASTROS_DBG_PRINT(verbose, "ASTROS_SPDK_TEST(%d)\n", 1);
}

ASTROS_DBG_PRINT(verbose, "ASTROS_SPDK_TEST TRADDR(%s)\n", g_trid.traddr);


ASTROS_DBG_PRINT(verbose, "g_trid.trtype = %d\n", g_trid.trtype);
ASTROS_DBG_PRINT(verbose, "g_trid.trstring = %s\n", g_trid.trstring);
ASTROS_DBG_PRINT(verbose, "g_trid.subnqn = %s\n", g_trid.subnqn);

if (strlen(g_trid.traddr) != 0) 
{
#if 0

	struct spdk_nvme_ctrlr_opts opts;

	spdk_nvme_ctrlr_get_default_ctrlr_opts(&opts, sizeof(opts));
	memcpy(opts.hostnqn, g_hostnqn, sizeof(opts.hostnqn));
	ctrlr = spdk_nvme_connect(&g_trid, &opts, sizeof(opts));
	if (!ctrlr) {
		fprintf(stderr, "spdk_nvme_connect() failed\n");
		rc = 1;
		goto exit;
	}

	g_controllers_found++;
//	print_controller(ctrlr, &g_trid, spdk_nvme_ctrlr_get_opts(ctrlr));
	spdk_nvme_detach_async(ctrlr, &g_detach_ctx);
#endif
} 
else if (spdk_nvme_probe(&g_trid, NULL, probe_cb, attach_cb, NULL) != 0) 
{
	fprintf(stderr, "spdk_nvme_probe() failed\n");
	rc = 1;
	goto exit;
}

	
exit:
	return 0;
}


#endif
