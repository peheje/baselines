import cProfile
import io
import pstats


class Profiler:
    """ Quick profiling of your code with "with" statement. """
    def __init__(self):
        self.pr = cProfile.Profile()

    def __enter__(self):
        self.pr.enable()
        return self.pr

    def __exit__(self, type, value, traceback):
        self.pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(self.pr, stream=s).sort_stats("cumulative")
        ps.print_stats()
        print(s.getvalue())
