

import os

def current_dir():
    abspath = os.path.abspath(__file__)
    print('absolute path: ' , abspath)

    dirname = os.path.dirname(abspath)
    print('directory name: ' , dirname)

    return dirname


def append_path(dirname, relpath):
    return os.path.join(dirname, relpath)


def extract_dir(filepath):
    drive, path_and_file = os.path.splitdrive(filepath)
    path, filename = os.path.split(path_and_file)
    return path, filename


def get_path(relpath):
    print('\n======= My directory ======')
    print('relative path: ' , relpath)
    dirname = current_dir()

    print('\n\n======= Target directory ======')
    filepath = append_path(dirname, relpath)
    print('file path: '     , filepath)
    path, filename = extract_dir(filepath)
    print('path: '          , path)
    print('file name: '     , filename)



