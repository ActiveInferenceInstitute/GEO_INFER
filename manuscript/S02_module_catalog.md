# Supplemental Module Catalog {#sec:module_catalog}

The catalog is the per-module companion to the theme map in
[@sec:system_context] and the measured inventory in
[@tbl:module_inventory]. Where those surfaces group the framework, this
catalog enumerates it: one prose entry per module, in alphabetical order,
each stating what the module is for, what its public interface exposes, and
how its test surface is shaped. The repository ships
`{{GEO_MODULE_COUNT}}` modules at the commit this manuscript was generated
from.

[@tbl:module_catalog_index] is the catalog's index. Each row names one
measured module, the declared theme it belongs to, a one-line purpose taken
from the module's own `README.md`, its counted non-empty Python source
lines, and its test-file count. The entries that follow the table expand
each row into prose; they are authored from the module's public interface —
package exports and explicitly named owning submodules — its verification surfaces,
and its declared theme role. Volatile source and test counts appear only in
this generated index rather than being copied into each authored entry.
These counts describe inventory, while command receipts record collection,
execution and results. Neither inventory size nor a module's theme establishes
method correctness, security, field calibration or operational effectiveness.
For current repository navigation and verification commands, use the
[public root README Module Index](https://github.com/ActiveInferenceInstitute/GEO_INFER/blob/main/README.md#module-index).
The source-stamped tables in this manuscript describe its measured checkout.

{{GEO_MODULE_TABLE}}

: The module catalog index at commit `{{RESEARCH_COMMIT}}`: one row per
measured module with its declared theme, purpose, and measured source and
test surfaces. The rows are alphabetical and self-contained, so a row can be
read on its own or as the index into the prose entries below. The purpose
column quotes the module's own README. LOC counts non-empty Python source
lines under `src/`, whereas the inventory figure and module inventory table
count source files. Tests counts Python test files, using the same definition
as those other inventory displays.
{#tbl:module_catalog_index}

Read the catalog alongside the theme map rather than instead of it: theme
membership is a statement about the kind of claim a module is responsible
for, while the entries below are statements about what each module actually
exposes and verifies. An entry can be read on its own — each begins with the
module's full name — or in sequence, where the alphabetical order makes the
shared shape of the module contract visible: every module is a package with
a public interface, a test tree, and a theme.
