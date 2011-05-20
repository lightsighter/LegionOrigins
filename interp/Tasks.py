import threading
import sys
import traceback

class UnhandledSubtaskException(Exception):
    def __init__(self, exc_info):
        self.exc_info = exc_info

class FutureValue(object):
    def __init__(self):
        self.condvar = threading.Condition()
        self.result_ready = False
        self.result_value = None
        self.result_except = None

    def is_ready(self):
        return self.result_ready # don't have to take lock here

    def get_result(self):
        with self.condvar:
            if not self.result_ready:
                self.condvar.wait()
        if self.result_except is not None:
            raise UnhandledSubtaskException(self.result_except)
        return self.result_value

    def set_result(self, value):
        with self.condvar:
            if self.result_ready:
                raise DuplicateFutureResult(self, value)
            self.result_ready = True
            self.result_value = value
            self.condvar.notify_all()

    def set_exception(self, exc_info):
        with self.condvar:
            if self.result_ready:
                raise DuplicateFutureResult(self, value)
            self.result_ready = True
            self.result_value = None
            self.result_except = exc_info
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
        self.exc_info = None

    def run(self):
        from Runtime import TaskContext
        TaskContext.set_current_context(self.context)
        try:
            rv = self.func(*self.args, **self.kwargs)
            self.result.set_result(rv)
        except UnhandledSubtaskException:
            # pass the exception upstream without comment
            e = sys.exc_info()
            self.result.set_exception(e)
        except:
            # all other exceptions get displayed right away
            e = sys.exc_info()
            traceback.print_exception(*e)
            self.result.set_exception(e)
