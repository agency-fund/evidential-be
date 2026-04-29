# This file intentionally left blank. This allows pytest's rootdir-walk behavior to treat `src` as a package; otherwise
# the conftest at src/xngin/apiserver/conftest.py is registered under both its file path and the dotted name
# `xngin.apiserver.conftest`, which collides with some of the pytest_plugins entries.
