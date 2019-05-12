
import os

def path(path):
    return os.path.abspath(os.path.expanduser(path))

def filename(filenames, tags):

    if not isinstance(filenames, list) or not isinstance(tags, list):
        raise ValueError("Both arguements must be instances of the 'list' object")

    filenames = sorted(set(map(path, filenames)))

    filepath = []

    for filename in filenames:
        
        filepath.append(os.path.splitext(os.path.basename(filename))[0])

    filepath = '_'.join(filepath + tags)

    return os.path.join(os.path.curdir, 'out', filepath)

