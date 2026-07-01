def parse_source_from_filename(file_path: Path, dataset_name: str) -> str:
    """
    Extract the ``source`` token from a results filename following the
    ``<dataset>_<source>.json`` convention.

    Falls back gracefully when the filename does not exactly match the
    dataset name prefix, by trying the substring after the first underscore.
    """
    stem = file_path.stem  # filename without .json
    prefix = f"{dataset_name}_"
    
    # Primary logic: if file is "CRE_azure", strip "CRE_" and return "azure"
    if stem.startswith(prefix) and len(stem) > len(prefix):
        return stem[len(prefix):]
        
    # Fallback: If it doesn't match perfectly but has an underscore, 
    # assume everything after the first underscore is the source.
    if "_" in stem:
        return stem.split("_", 1)[1]
        
    return stem or "unknown"
