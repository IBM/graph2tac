import pytest

# pytest overwrites assert statements in test
# to do the same for testing code in other files, we have to
# register those modules before they get imported.
# This turns all assert statements found under tests
# into pytest assert statements.
pytest.register_assert_rewrite('tests')
