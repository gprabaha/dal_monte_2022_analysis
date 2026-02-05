# scripts

Scripts are thin wrappers around package functions. They provide quick entry points
for batch processing without hard-coding too much logic in the script itself.

If a script grows beyond a few lines, move the logic into `src/` and keep the script
as a simple CLI shim.

Plotting scripts live in `scripts/plotting/` and use shared styling from
`configs/plotting.yaml`.
