"""`scripts/` is not a package — the deploy runs its modules as scripts — so the
directory goes on the path here rather than being imported from an installed
name."""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
