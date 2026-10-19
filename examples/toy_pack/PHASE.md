# Toy pack template

## Premise

This two-row fixture checks pack validation, ordinary workflow execution,
catalog discovery and local reporting. It does not test a model hypothesis.
The deliberately false fixture assertion “three rows were mapped” exercises a
refuted verdict after the two-row result is observed. It earns engineering
coverage only. Actual studies use their own question and record format.
Do not copy fixture outcomes into study claims.

## Artifacts

- `workflow.py`: executable workflow source.
- `pack.json`: versioned metadata and recipe/source locks.
- `README.md`: validation, execution and discovery commands.
