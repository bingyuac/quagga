class Connector(object):
    def __init__(self, matrix, forward_context):
        self.matrix = matrix
        self.forward_context = forward_context
        self.backward_context = None
        self.updated = False
        self._derivatives = []
        self._backward_contexts = []

    def register_derivative(self, dL_dmatirx, backward_context):
        if not self.matrix.same_shape(dL_dmatirx):
            raise ValueError('Derivative has different shape:({}, {}) '
                             'from variable shape:({}, {})'.
                             format(dL_dmatirx.nrows, dL_dmatirx.ncols,
                                    self.matrix.nrows, self.matrix.ncols))

        self._derivatives.append(dL_dmatirx)
        self._backward_contexts.append(backward_context)
        if not self._derivatives:
            self.backward_context = self._backward_contexts[0]

    @property
    def derivative(self):
        if self.updated:
            self.backward_context.depend_on(*self._backward_contexts[1:])
            for derivative in self._derivatives[1:]:
                self._derivatives[0].add(self.backward_context, derivative)
            self.updated = False
        return self._derivatives[0]

    def __getattr__(self, name):
        attribute = getattr(self.matrix, name)
        if hasattr(attribute, '__call__'):
            def update_tracking_attribute(*args, **kwargs):
                self.updated = True
                return attribute(*args, **kwargs)
            setattr(self, name, update_tracking_attribute)
        else:
            def fget(self):
                self.updated = True
                return getattr(self.matrix, name)

            def fset(self, value):
                self.updated = True
                setattr(self.matrix, name, value)

            setattr(self, name, property(fget, fset))
        return attribute
