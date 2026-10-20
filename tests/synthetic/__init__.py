"""
Data generation, not assertion. These modules don't test.

The modules under tests/ import across this boundary, and these don't import
back out of it. That keeps the split meaningful.
"""
