config = {'processor_type': 'gpu'}


def get_processors_types():
    for processor_type in ['cpu', 'gpu']:
        config['processor_type'] = processor_type
        yield processor_type