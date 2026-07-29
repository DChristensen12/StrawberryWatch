# Deliberately no re-exports. Importing the package should not drag in torch,
# so callers reach for the submodule they actually want.
__version__ = "0.1.0"
