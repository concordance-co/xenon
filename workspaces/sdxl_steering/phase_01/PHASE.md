# SDXL steering check

The runnable workflow consumes phase 00’s direction and compares a neutral baseline with +4, +12 and −4 RMS edits using paired noise. It saves images, bounded diagnostics and a local gallery.

The engineering checks demonstrated tensor/image changes, including distortion under strong edits. They do not establish clean semantic control or perceptual quality. Larger held-out studies and human ratings are separate research work.

Use [the recipe guide](../README.md) and `workflow.py` for current execution instructions. Record new experiment outcomes in a new phase rather than extending this fixture’s old run history.
