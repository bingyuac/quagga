processor_type = 'gpu'
dtype = 'float'


def get_processors_types():
    for each in ['cpu', 'gpu']:
        global processor_type
        processor_type = each
        yield each