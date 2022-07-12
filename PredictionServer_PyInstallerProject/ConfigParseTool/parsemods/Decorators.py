# singleton Decorator
# @details singleton provides a decorator that classes can utilize for singleton functionality.
#
def singleton(coreClass):

    ## A class variable decorator for a class to make a singleton out of it
    _classInstances = {}

    ## getInstance
    # @details Default behavior for a singleton object to call this function to create
    # or retrieve only a single instance of the object
    # @params args
    # @params kwargs
    # @return the object instance of a class
    #
    def getInstance(*args, **kwargs):
        import os
        if os.environ.__contains__('PYTEST_CURRENT_TEST'):
            key = (coreClass, args, repr(kwargs), os.environ.get('PYTEST_CURRENT_TEST').split(' ')[0])
        else:
            key = (coreClass, args, repr(kwargs))
        if key not in _classInstances:
            _classInstances[key] = coreClass(*args, **kwargs)
        return _classInstances[key]

    return getInstance
