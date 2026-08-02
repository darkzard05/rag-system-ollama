# Ultrawork Notepad — Fix Main Page Height Calculation
Started: 2026-07-02T17:24:15Z

## Plan (exhaustive, atomic)
TBD

## Scenarios (the contract)
1. Happy Path: Viewport height is correctly detected -> Target height calculated -> Both PDF and Chat containers have identical heights -> No page-level overflow.
2. Edge Case: Viewport height is None (first render) -> Default height applied -> Layout remains stable -> Updates once JS returns.
3. Regression: Sidebar expanded/collapsed does not break the height calculation of the main columns.

## Now (single step in progress)
- Initial analysis complete. Preparing to invoke Plan Agent.

## Todo (remaining, ordered)
- [ ] Invoke Plan Agent to design the fix
- [ ] Implement height constraint in render_pdf_column
- [ ] Refine target_height calculation in _render_app_layout
- [ ] Verify with visual-qa
