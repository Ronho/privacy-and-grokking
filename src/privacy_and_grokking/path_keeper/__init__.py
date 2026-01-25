from .keeper import PathKeeper

_PATH_KEEPER = PathKeeper()


def get_path_keeper() -> PathKeeper:
    return _PATH_KEEPER


__all__ = ["PathKeeper", "get_path_keeper"]
