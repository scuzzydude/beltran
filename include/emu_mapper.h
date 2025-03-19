#ifndef __EMU_MAPPER_H
#define __EMU_MAPPER_H


typedef union
{
	nvm_cmd_t    nvme_cmd;
	//reserve for new command formats for storage next
	//assume that they will be smaller or equal to 64-byte NVMe (which is overkill)

} storage_next_command;

typedef struct 
{
	storage_next_command cmd;
	//Used by each level or mapped as an implementation structure
	//or a pointer holder if mapped to device/emulator managed memory
	uint64_t    storage_implementation_context[8];  
} storage_next_emuluator_context;

#define EMU_CONTEXT storage_next_emuluator_context
#define EMU_COMPONENT_NAME_LEN 64


typedef struct 
{
	
	char szModelName[EMU_COMPONENT_NAME_LEN];

	
} emu_target_model;





typedef struct 
{
	char szMapName[EMU_COMPONENT_NAME_LEN];
	
	
} emu_mapper;



#endif /* __EMU_MAPPER_H */
