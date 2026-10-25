# runners

Inference backends behind the `Runner` interface, the one seam between the surfaces (server, CLI) and the engines.

The interface is in `base.py` and the registry is in `__init__.py`; each runner's docstring
describes its backend. How to add one: [../DESIGN_AND_LIMITATIONS.md](../DESIGN_AND_LIMITATIONS.md).
