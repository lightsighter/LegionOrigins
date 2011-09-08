
#ifndef __PARALLAX_RUNTIME__
#define __PARALLAX_RUNTIME__

#include <vector>
#include <set>
#include <map>

// Processor ID
typedef unsigned int pxProcID_t;
// Processor Set
typedef std::set<pxProcID_t> pxProcSet_t;
// Event Handle
typedef unsigned int pxEventHandle_t;
// Function Handle
typedef unsigned int pxFuncHandle_t;
// Logical Region ID
typedef unsigned int pxLogHandle_t;
// Physcial Region ID
typedef unsigned int pxPhyHandle_t;
// Type ID
typedef unsigned int pxTypeID_t;

// The coherence properties supported by the runtime
enum CoherenceProperties
{
	Read,
	Write,
	Allocate,
	Free,
	Snapshot,
};

/**
 * pxFunc_t
 * 
 * This is a function which is a generic template for a task.
 * A function must be compiled into a closure before it
 * can be executed by the runtime.
 *
 * @field numArgs - the number of arguments the function takes
 * @field functionHandle - the handle that the runtime uses to reference the function
 * @field function - the function pointer to the code
 */
typedef struct {
	unsigned int numArgs;
	pxFuncHandle_t functionHandle;
	void (*function)(...);
} pxFunc_t;

/**
 * pxClosure_t 
 *
 * The compiled version of a task.  Can either be generated
 * directly by the compiler or can be JIT-ed from a function
 * by the runtime.
 *
 * @field args - pointers to the arguments for the function
 * @field functionHandle - the functionHandle that the closure uses
 * @field procs - the set of processors the closure will be run over
 */
typedef struct {
	std::set<void*> args;
	pxFuncHandle_t functionHandle;	
	pxProcSet_t *procs;
} pxClosure_t;


/**
 * LogicalRegion
 *
 * A class for tracking a logical region in the runtime.
 *
 * @field handle - the runtime handle for the logical region
 * @field parent - the runtime handle for the parent logical region (if applicable)
 * @field type - the type-id of the elements contained in the region
 */
class LogicalRegion
{
public:
	pxLogHandle_t handle;	
	pxLogHandle_t parent;
	pxTypeID_t type;
};

/**
 * PhysicalRegion
 *
 * A class for tracking physical regions in the runtime
 *
 * @field handle - the runtime handle for the physical region
 * @field logical - the handle for the logical region of which this is an instance
 * @field _ptr - the pointer to the memory for the region
 * @field next_element - the next free space
 */
template<typename T>
class PhysicalRegion
{
public:
	pxPhyHandle_t handle;
	pxLogHandle_t logical;
	T *_ptr;
	unsigned int next_element;
public:
	T* allocate();
	void free(T *elem);
	void set(T* ptr, T value);
	T get(T *ptr)
};


class pxRuntime
{
// Runtime initialization
public:
	/**
	 * pxRuntime
	 *
	 * Create an instance of a pxRuntime
	 */
	pxRuntime();

	/**
	 * pxInitialize
	 *
	 * Initialize a pxRuntime, making it aware of the hardware it is responsible
	 * for as well as it's surrounding runtimes
	 *
	 * @param desc - description of the part of the machine the runtime manages
	 * @param parent - the parent runtime (can be null)
	 * @param children - the child runtimes, can be empty
	 */
	virtual void pxInitialize(MachineDescription m, 
					pxRuntime *parent, 
					std::vector<pxRuntime*> children) = 0;

	/**
	 * pxFinialize
	 *
	 * perform any operations to shut down the runtime
	 */
	virtual void pxFinalize() = 0;

	/**
	 * pxWaitForEvent
	 * 
	 * Wait for an event handle to complete execution 
	 *
	 * @param handle - the event handle to wait for
	 */
	virtual void pxWaitForEvent(pxEventHandle_t handle) = 0;

//////////////////////////////////////////////////////
//     Operations for managing logical regions 
//////////////////////////////////////////////////////
public:
	/**
	 * pxCreateLogicalRegion
	 *
	 * Create a logical region with optional parent logical region
	 *
	 * @param handle - a reference to a place to assign the logical region value
	 * @param parent - the logical parent of the region to be create (if exists)
	 * @return - handle for the event
	 */
	virtual pxEventHandle_t pxCreateLogicalRegion(pxLogHandle_t &handle, 
							pxLogHandle_t parent = 0) = 0;

	/**
	 * pxDestroyLogicalRegion
	 *
	 * Destroy the specified logical region
	 *
	 * @param handle - the handle of the logical region to be destroyed
	 * @return - handle for the event
	 */
	virtual pxEventHandle_t pxDestroyLogicalRegion(pxLogHandle_t handle) = 0;

	/**
	 * pxAcquireLogicalRegion
	 *
	 * Acquire a logical region with the given coherence properties
	 *
	 * @param handle - the handle of the logical region to acquire
	 * @param prop - the set of coherence properties to obtain for the region
	 * @return - handle for the event
	 */
	virtual pxEventHandle_t pxAcquireLogicalRegion(pxLogHandle_t handle, 
							CoherenceProperties prop) = 0;

	/**
	 * pxReleaseLogicalRegion
	 * 
	 * Release the logical region from the given coherence properties
	 *
	 * @param handle - handle for the logical region to release
	 * @param prop - the properties to be release (can relase subsets)
	 * @return - handle for the event
	 */
	virtual pxEventHandle_t pxReleaseLogicalRegion(pxLogHandle_t handle, 
							CoherenceProprties prop) = 0;

	/**
	 * pxGetLogicalRegion
	 *
	 * Return a local pointer to the logical region
	 *
	 * @param handle - the handle for the logical region to find
	 * @param reg - a place to put a pointer to a logical region
	 * @return - a handle for the event
	 */
	virtual pxEventHandle_t pxGetLogicalRegion(pxLogHandle_t handle,
							LogicalRegion **reg) = 0;

//////////////////////////////////////////////////////
//     Operations for managing physical regions 
//////////////////////////////////////////////////////
public:
	/**
	 * pxCreatePhysicalRegion
	 *
	 * Create a physical instance of a logical region
	 *
	 * @param phy_handle - a reference to a handle for a physical region
	 * @param log_handle - the handle of a logical region to create an instance of 
	 * @return - handle for the event
	 */
	virtual pxEventHandle_t pxCreatePhysicalRegion(pxPhyHandle_t &phy_handle, 
							pxLogHandle_t log_handle) = 0;

	/**
	 * pxDestroyPhysicalRegion
	 *
	 * Destroy the specified physical instance of a region
	 * 
	 * @param handle - a handle for the physical region to destroy
	 * @return - handle for the event
	 */
	virtual pxEventHandle_t pxDestroyPhysicalRegion(pxPhyHandle_t handle) = 0;

	/** 
	 * pxCopyPhysicalRegion
	 *
	 * Copy one physical region to another, at least one physical region should be local
	 *
	 * @param dst_handle - handle for the destination physical region
	 * @param src_handle - handle for the source physcial region
	 * @return - handle for the event
	 */	
	virtual pxEventHandle_t pxCopyPhysicalRegion(pxPhyHandle_t dst_handle, 
							pxPhyHandle_t src_handle) = 0;

	/**
	 * pxGetPhysicalRegion	
	 *
	 * Return a local pointer to the specified physical region
	 *
	 * @param handle - the handle of the region to get a pointer to
	 * @param reg - a place to put a pointer to the region
	 * @return - a handle for the event
	 */
	template<typename T>
	virtual pxEventHandle_t pxGetPhysicalRegion(pxPhyHandle_t handle,
							PhysicalRegion<T> **reg) = 0;
	
//////////////////////////////////////////////////////
//     Operations for JIT compilation and creating parallel work 
//////////////////////////////////////////////////////
public:
	/**
	 * pxRegisterFunction
	 *
	 * Register a function (an not-yet-compiled closure) with the runtime
	 *
	 * @param function - the function to be registered with the runtime
	 * @return - handle for the event
	 */
	virtual pxEventHandle_t pxRegisterFunction(pxFunction_t function) = 0;	

	/**
	 * pxJITCompileFunction
	 *
	 * JIT compile a specific function based on the available resources
	 *
	 * @param handle - the function handle to compile
	 * @param closure - a place to store the pointer to the closure
	 * @return - a handle for the event
	 */
	virtual pxEventHandle_t pxJITCompile(pxFuncHandle_t handle, 
						pxClosure_t **closure) = 0;

	/**
	 * pxFreeClosure
	 *
	 * Free a runtime created closure
	 * 
	 * @param closure - a pointer to the closure to be freed
	 * @return - a handle for the event
	 */
	virtual pxEventHandle_t pxFreeClosure(pxClosure_t *closure) = 0;

	/**
	 * pxLaunchClosure
	 *
	 * Execute a closure 
	 *
	 * @param closure - the closure to be executed
	 * @return - a handle for the event
	 */
	virtual pxEventHandle_t pxLaunchClosure(pxClosure_t *closure) = 0;
};

#endif // __PARALLAX_RUNTIME__

