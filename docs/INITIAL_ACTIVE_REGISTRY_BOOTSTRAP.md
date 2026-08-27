# Initial ACTIVE registry bootstrap

`publish_initial_active_registry.py` is a one-time migration utility, not a normal
release-processing command. It can only initialize an empty registry and requires
`--acknowledge-initial-migration`. It refuses to overwrite any existing registry.

Normal releases must use the control-plane lifecycle:

`DETECTED -> ACQUIRED -> VALIDATED -> ACTIVE`

Promotion is explicit and fail-closed. Failed or merely validated candidates never
replace ACTIVE. Use `--verify-existing` for a read-only bootstrap audit.
