# This file intentionally left blank. It makes pytest's rootdir-walk treat `src` as a package, so test modules and
# conftests are imported as `src.xngin...`. Don't import a conftest as a module or name one in pytest_plugins: that
# would load a second copy of it, registering its fixtures and hooks twice.
