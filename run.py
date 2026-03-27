"""
run.py
======
Institutional-grade entry point for the Sovereign Engine.
Redirects to screener_v14_modular.py.
"""

import sys
import screener_v14_modular

if __name__ == "__main__":
    # screener_v14_modular.py handles its own CLI parsing and logging
    # upon import, but the actual execution loop depends on it being
    # run as a script or having a main() called.
    
    # We check if screener_v14_modular has a main() or if we should just
    # let its module-level execution handle it (current v14 design).
    # Since screener_v14_modular.py executes on import (due to setup_logging 
    # and main block calls), simply importing it runs the engine.
    pass
