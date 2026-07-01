def split_model_key(model_key: str, source: str) -> tuple[str, list[str]]:
    """
    Best-effort split of a model key into a base ``model_name`` and a list
    of recognized ``features`` (see ``KNOWN_FEATURE_TOKENS``).
    Scans all tokens so features are found regardless of their position.
    """
    tokens = model_key.split("_")
    
    # Strip the source prefix if it's there
    if tokens and tokens[0].lower() == source.strip().lower():
        tokens = tokens[1:]
        
    if not tokens:
        return model_key, []

    features: list[str] = []
    remaining_tokens: list[str] = []
    
    # Check every single token
    for token in tokens:
        candidate = token.lower()
        if candidate in KNOWN_FEATURE_TOKENS:
            feat = KNOWN_FEATURE_TOKENS[candidate]
            if feat not in features:  # Prevent duplicate tags
                features.append(feat)
        else:
            remaining_tokens.append(token)
            
    model_name = "_".join(remaining_tokens) if remaining_tokens else model_key
    
    return model_name, features
