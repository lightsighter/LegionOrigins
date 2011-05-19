import threading
import sys

class FutureValue(object):
    def __init__(self):
        self.condvar = threading.Condition()
        self.result_ready = False
        self.result_value = None

    def is_ready(self):
        return self.result_ready # don't have to take lock here

    def get_result(self):
        with self.condvar:
            if not self.result_ready:
                self.condvar.wait()
        return self.result_value

    def set_result(self, value):
        with self.condvar:
            if self.result_ready:
                raise DuplicateFutureResult(self, value)
            self.result_ready = True
            self.result_value = value
            self.condvar.notify_all()

class TaskThread(threading.Thread):
    def __init__(self, func, args, kwargs, context):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.context = context
        context.task = self
        self.result = FutureValue()

    def run(self):
        from Runtime import TaskContext
        TaskContext.set_current_context(self.context)
        try:
            rv = self.func(*self.args, **self.kwargs)
            self.result.set_result(rv)
        except:
           e = sys.exc_info()
           print "Unhandled_exception in thread: ", str(e)
           self.result.set_result(e)
