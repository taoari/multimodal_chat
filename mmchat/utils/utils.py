from functools import wraps
import inspect

def change_signature(arg_list, kwarg_dict={}):
    def decorator(fn):
        # Create a signature from arg_list and kwarg_dict
        parameters = []
        for arg in arg_list:
            parameters.append(inspect.Parameter(arg, inspect.Parameter.POSITIONAL_OR_KEYWORD))

        for kwarg, default in kwarg_dict.items():
            parameters.append(inspect.Parameter(kwarg, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default))

        new_signature = inspect.Signature(parameters)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            bound_args = new_signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            result = fn(*bound_args.args, **bound_args.kwargs)
            
            # Handle generator functions
            if inspect.isgeneratorfunction(fn):
                yield from result
            else:
                return result

        wrapper.__signature__ = new_signature
        return wrapper
    return decorator