"""Compatibility with checkpoints saved before shared spectral geometry."""


def compatible_state_dict(state, model):
    """Rename historical fixed buffers while retaining strict parameter checks."""
    expected = model.state_dict()
    migrated = state.copy()
    geometry_buffers = {
        "basis", "exposure_profile", "spectral_eigenvalues",
        "local_smoothness_weights", "roughness",
    }
    for key in state:
        prefix, separator, name = key.rpartition(".filter_fn.")
        if not separator or name not in geometry_buffers or key in expected:
            continue
        replacement = prefix + ".filter_fn.spectral_geometry." + name
        if replacement in expected:
            migrated[replacement] = migrated.pop(key)
        elif name == "roughness" and prefix + ".filter_fn.spectral_geometry.basis" in expected:
            # Hard truncation uses the saved basis. Older versions also stored
            # the unused dense roughness matrix in each filter.
            migrated.pop(key)
    if hasattr(state, "_metadata"):
        migrated._metadata = state._metadata
    return migrated
