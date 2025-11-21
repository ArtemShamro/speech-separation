class DummyWriter:
    def __getattr__(self, name):
        def dummy(*args, **kwargs):
            return None

        return dummy
