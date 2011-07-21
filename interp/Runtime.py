import threading
from Regions import Region
from Tasks import TaskThread, FutureValue

class Runtime(object):
    def __init__(self, machine_desc):
        self.machine_desc = machine_desc

    def run_application(self, main_func, *main_args, **main_kwargs):
        '''start the main thread for an application on an arbitrary processor'''
        main = TaskThread(main_func, main_args, main_kwargs,
                          TaskContext(self, None, "foo"))
        main.start()
        return main.result.get_result()

    def run_task(self, task_func, *task_args, **task_kwargs):
        '''spawn a new task, specifying which regions are needed, with optional mapping hints'''
        # see which regions the task needs
        from Regions import get_task_regions
        regions_needed = get_task_regions(task_func, task_args, task_kwargs)
        # TODO: actual mapping
        location = "foo"
        newctx = TaskContext(self, None, location)

        for r in regions_needed:
            # make sure the current context has sufficient access to the region as well
            b = TaskContext.get_region_binding(r["region"], min_access = r["mode"])

            newctx.add_region_bindings(RegionBinding(r["region"],
                                                     r["region"].get_instance(location),
                                                     r["mode"]))
        newthr = TaskThread(task_func, task_args, task_kwargs, newctx)

        newthr.start()
        # in debug mode, wait for child thread to finish
        do_debug = True
        if do_debug:
            newthr.join()
            newthr.result.get_result()  # if the subtask had an exception, so will this
        return newthr.result  # this is a future, not the actual result

############################################################

class RegionBinding(object):
    def __init__(self, logical_region, phys_inst, mode):
        self.logical_region = logical_region
        self.phys_inst = phys_inst
        self.mode = mode

    def __repr__(self):
        return "BIND(" + repr(self.logical_region) + " -> " + repr(self.phys_inst) + " : " + str(self.mode) + ")"


class UnmappedRegionException(Exception):
    def __init__(self, context, logical_region, exact, min_access):
        self.context = context
        self.logical_region = logical_region

    def __str__(self):
        return "Region (" + str(self.logical_region) + ") unmapped in:" + (''.join(map(lambda b: "\n  " + str(b), self.context.bindings)) if len(self.context.bindings) > 0 else "NO BINDINGS")

############################################################


class TaskContext(object):
    thread_local_storage = threading.local()

    def __init__(self, runtime, task, processor, bindings = None):
        self.runtime = runtime
        self.task = task
        self.processor = processor
        self.bindings = bindings if bindings is not None else []
        pass

    def __str__(self):
        return "[Ctx: %s @ %s (%s)]" % (self.task,
                                        self.processor,
                                        ",".join(map(str, self.bindings)))

    @classmethod
    def get_current_context(self): return TaskContext.thread_local_storage.context

    @classmethod
    def set_current_context(self, new_ctx): TaskContext.thread_local_storage.context = new_ctx
    
    # all of the methods below are defined as class methods - ideally they'd be
    #   both class and instance methods so that we can detect erroneous cases
    #   in which the wrong (i.e. not yours) context is used
    @classmethod
    def get_runtime(self):
        '''returns a pointer to the global runtime object'''
        self = self.get_current_context()
        return self.runtime

    @classmethod
    def get_task(self):
        '''returns the current task'''
        self = self.get_current_context()
        return self.task

    @classmethod
    def get_processor(self):
        '''returns the processor the task is running on'''
        self = self.get_current_context()
        return self.processor

    # not classmethod
    def add_region_bindings(self, *args):
        '''adds region bindings to the current context'''
        #self = self.get_current_context()
        self.bindings.append(*args)

    @classmethod
    def get_region_binding(self, logical_region, exact = False, must_match = True, min_access = None):
        '''returns the region binding being used for a logical region'''
        self = self.get_current_context()

        # try for exact matches first (not sure this matters)
        for b in self.bindings:
            if (min_access is not None) and not(min_access.is_subset_of(b.mode)): continue
            if b.logical_region == logical_region: return b

        # now allow bindings of supersets of the region we want
        if not exact:
            for b in self.bindings:
                if (min_access is not None) and not(min_access.is_subset_of(b.mode)): continue
                if logical_region.is_subset_of(b.logical_region):
                    return RegionBinding(logical_region, b.phys_inst, b.mode)

        if must_match:
            raise UnmappedRegionException(self, logical_region, exact, min_access)
        return None


############################################################

