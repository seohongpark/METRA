import copy

_g_session = None
_g_context = {}


class GlobalContext:
    def __init__(self, context):
        self.context = context

    def __enter__(self):
        global _g_context
        self.prev_g_context = _g_context
        _g_context = self.context

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _g_context
        _g_context = self.prev_g_context


def get_metric_prefix():
    global _g_context
    prefix = ''
    if 'phase' in _g_context:
        prefix += _g_context['phase'].capitalize()
    if 'policy' in _g_context:
        prefix += {'sampling': 'Sp', 'option': 'Op'}.get(
            _g_context['policy'].lower(), _g_context['policy'].lower()).capitalize()

    if len(prefix) == 0:
        return '', ''

    return prefix + '/'


def get_context():
    global _g_context
    return copy.copy(_g_context)
