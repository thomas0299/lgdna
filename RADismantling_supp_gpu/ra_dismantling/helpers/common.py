# https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
def product_dict(_callback=None, **kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()

    from itertools import product

    for instance in product(*vals):
        instance = dict(zip(keys, instance))

        if _callback:
            instance = _callback(instance)

            if not instance:
                continue

        yield instance
