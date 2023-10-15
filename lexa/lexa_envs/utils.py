def parse(spec, options):
  """Parses a spec string given a list of options. This can use parametric options, e.g.
  parse(s, ['targetrew(\d*)x']). They can be accessed as e.g. targetrewNx.

  :param spec: a string separated by underscore, e.g. kitchen_microwave_fullstaterev_topviewcent.
  It will be parsed to determine whether any of the spec elements match an option.
  :param options: a list of regular expressions that will be matched against the spec elements. They will be matched  exactly.
  :return: a dictionary of truth values for each option, and a spec string with all parsed elements removed.
  """

  opt_names = [o.replace('(\d*)', 'N') for o in options]
  options = [re.compile('^' + o + '$') for o in options]

  # Parse
  spec_list = [s for s in spec.split('_') if any(o.search(s) for o in options)]
  def parse_option(o):
    match = any_obj(o.match(s) for s in spec_list)
    if not match:
      return False
    if len(match.groups()) == 0:
      return True
    return match.groups()[0]
  matches = [parse_option(o) for o in options]
  keys = dict((n, m) for n, m in zip(opt_names, matches))

  # Remove keys from name
  spec = '_'.join([s for s in spec.split('_') if s not in spec_list])
  return keys, spec


def any_obj(l):
  l = list(l)
  nonzero = np.nonzero(l)[0]
  if len(nonzero) > 0:
    return l[nonzero[0]]
  else:
    return False


def translate(n, dictionary):
  # Get key values
  for key, value in dictionary.items():
    n = n.replace(key, value)
  return n

def subdict(dict, keys, strict=True):
  if not strict:
      keys = dict.keys() & keys
  return AttrDict((k, dict[k]) for k in keys)