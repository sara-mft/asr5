import re

def compute_diarization_metrics(
    reference_text: str, 
    hypothesis_segments: list['DiarizationSegment']
) -> dict[str, float]:
    """
    Parses a conversational reference text and computes DER and JER against
    the engine's hypothesis segments.
    """
    try:
        from pyannote.core import Annotation, Segment
        from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
    except ImportError as e:
        raise RuntimeError("pyannote.metrics not installed. Run: pip install pyannote.metrics") from e

    # 1. Parse Reference Text
    # Regex matches: [00:01.4 - 00:04.9]  Guest-1: Madame Morel et venue.
    line_re = re.compile(r"\[(\d{2}):(\d{2}\.\d+)\s*-\s*(\d{2}):(\d{2}\.\d+)\]\s*([^:]+):")
    
    ref_annotation = Annotation()
    for line in reference_text.splitlines():
        match = line_re.search(line)
        if match:
            m1, s1, m2, s2, speaker = match.groups()
            start_sec = int(m1) * 60 + float(s1)
            end_sec = int(m2) * 60 + float(s2)
            ref_annotation[Segment(start_sec, end_sec)] = speaker.strip()

    # 2. Build Hypothesis Annotation
    hyp_annotation = Annotation()
    for seg in hypothesis_segments:
        hyp_annotation[Segment(seg.start_sec, seg.end_sec)] = seg.speaker

    # 3. Calculate Metrics
    if not ref_annotation.labels():
        return {"der": 0.0, "jer": 0.0}

    der_metric = DiarizationErrorRate(collar=0.250)
    jer_metric = JaccardErrorRate()

    return {
        "der": round(der_metric(ref_annotation, hyp_annotation), 4),
        "jer": round(jer_metric(ref_annotation, hyp_annotation), 4),
    }
