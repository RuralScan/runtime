"""Data adapters. They fetch and convert layers; they never decide analytical rules.

variables_loader  load any catalog variable onto the grid, from a file or Earth Engine, with a cache
gee               Earth Engine
hf                files: the Hugging Face dataset, a local copy, or a remote raster
synthetic         offline test data with plausible ranges (never real results)
"""
