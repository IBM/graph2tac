import pytest

# pytest overwrites assert statements in each test
# to do the same for testing code in helper files, we have to
# register those modules before they get imported.
# This turns all assert statements in helper code
# in the integration and unit modules into pytest assert statements.
pytest.register_assert_rewrite('tests.integration')
pytest.register_assert_rewrite('tests.unit')
