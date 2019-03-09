
class APIServiceUndefined(Exception):
    """Happens when an undefined API service is used in global conf"""
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)
