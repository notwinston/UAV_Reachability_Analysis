"""OptimizedDP-compatible HJ solver — pure NumPy implementation.

Implements the same algorithms as SFU-MARS/optimized_dp but without the
HeteroCL dependency. Works on Python 3.10+.

Key components:
- Grid: computational grid (same API as odp.Grid.GridProcessing.Grid)
- HJSolver: backward reachable set/tube solver
- computeSpatDerivArray: spatial derivative computation
"""
