def remove_prefix(string, prefix):
    return string[len(prefix):]if string.startswith(prefix) else string[:]


def remove_suffix(string, suffix):
    return string[:-len(suffix)] if (suffix and string.endswith(suffix)) else string[:]