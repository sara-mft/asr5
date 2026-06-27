"""
metrics.py — Transcription Accuracy Metrics  [v3]
==================================================
ASR evaluation metrics.

v3 — corrige 11 problèmes identifiés lors de l'audit de v2 :
  #1  Diarisation : ne renvoie plus un score parfait silencieux sur échec de parsing.
  #2  CER : cohérente avec compute_wer sur le cas (référence vide, hypothèse non vide).
  #3  Ponctuation macro_f1 : exclut les classes sans support au lieu de les noter 0.0.
  #4  normalise_numbers : corrige le bug \\b qui désactivait €, $, £, % ; préserve la devise réelle.
  #5  MER : dénominateur conforme à Morris, Maier & Green (2004), N+I au lieu de max(N,P).
  #6  WIP/WIL : cohérente avec compute_wer sur le cas (référence ET hypothèse vides).
  #7  Ponctuation : punct_class() ancrée en fin de mot (ne confond plus virgule décimale et ponctuation).
  #8  normalise_numbers : composition arithmétique réelle des nombres FR (au lieu d'une substitution mot-à-mot).
  #9  Suppression du code mort (anciennes formules commentées).
  #10 Nouvelles fonctions : aggregate_wer_corpus() (WER poolée) et to_json_safe() (sérialisation sans inf/NaN).
  #11 Nouvelle fonction : compute_term_metrics() — précision/rappel/F1 au niveau des termes métier,
      complète compute_tner() qui est aveugle aux termes hallucinés.
"""

from __future__ import annotations

import re
import unicodedata
import warnings
from collections import Counter

# ──────────────────────────────────────────────────────────────────────────────
# Text normalisation
# ──────────────────────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    """Lowercase, strip punctuation (keep apostrophes), collapse spaces."""
    text = unicodedata.normalize("NFC", text).lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenise(text: str) -> list[str]:
    return normalise(text).split()


# ──────────────────────────────────────────────────────────────────────────────
# Number normalisation (for NormWER) — FIX #4 et #8
# ──────────────────────────────────────────────────────────────────────────────

_FUSED_THOUSANDS = re.compile(r"(\d)\s(\d{3})(?!\d)")

# FIX #4 : (?!\w) au lieu de \b — fonctionne aussi après un symbole non-mot (€, $, £, %),
# alors que \b ne peut jamais matcher juste après un caractère non-mot.
_CURRENCY_MAP = {
    "€": "eur", "$": "usd", "£": "gbp",
    "usd": "usd", "eur": "eur", "gbp": "gbp",
    "euros": "eur", "euro": "eur",
    "dollars": "usd", "dollar": "usd",
    "livres": "gbp", "livre": "gbp",
}
_CURRENCY_SYMBOLS = re.compile(
    r"(\d)\s*(€|\$|£|usd|eur|gbp|euros?|dollars?|livres?)(?!\w)", re.IGNORECASE
)
_PERCENT = re.compile(r"(\d)\s*(%|pourcent|pour\s+cent)(?!\w)", re.IGNORECASE)
_DATE_SEP = re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b")

# FIX #8 : vocabulaire numéral FR utilisé par le parseur compositionnel ci-dessous.
_FR_NUM_UNITS = {
    "zéro": 0, "zero": 0, "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4,
    "cinq": 5, "six": 6, "sept": 7, "huit": 8, "neuf": 9, "dix": 10, "onze": 11,
    "douze": 12, "treize": 13, "quatorze": 14, "quinze": 15, "seize": 16,
}
_FR_NUM_TENS = {
    "vingt": 20, "trente": 30, "quarante": 40, "cinquante": 50, "soixante": 60,
    "septante": 70, "huitante": 80, "octante": 80, "nonante": 90,
}
_FR_NUM_MULT = {"cent": 100, "cents": 100}          # multiplie le groupe courant
_FR_NUM_THOUSAND = {"mille": 1000}                   # finalise le groupe x1000 dans le total
_FR_NUM_SCALE = {"million": 10**6, "millions": 10**6, "milliard": 10**9, "milliards": 10**9}
_FR_NUM_VOCAB = (
    set(_FR_NUM_UNITS) | set(_FR_NUM_TENS) | set(_FR_NUM_MULT)
    | set(_FR_NUM_THOUSAND) | set(_FR_NUM_SCALE) | {"vingts"}
)


def _parse_number_run(words: list[str]) -> int | None:
    """Résout une séquence de sous-mots numéraux FR déjà reconnus en un entier.
    Gère : unités, dizaines régulières, la construction vicésimale
    quatre-vingt(s)/quatre-vingt-dix, cent(s), mille, million(s), milliard(s),
    et le connecteur 'et' (vingt-et-un, soixante-et-onze).
    Portée documentée : 0 à quelques milliards, style "France" standard.
    Ne couvre pas : nombres ordinaux, septante/huitante/nonante combinés à des
    composés irréguliers, formes belges/suisses au-delà des dizaines simples.
    """
    total = 0
    group = 0
    i, n = 0, len(words)
    while i < n:
        w = words[i]
        if w == "et":
            i += 1
            continue
        if w == "quatre" and i + 1 < n and words[i + 1] in ("vingt", "vingts"):
            group += 80
            i += 2
            continue
        if w in _FR_NUM_UNITS:
            group += _FR_NUM_UNITS[w]
            i += 1
            continue
        if w in _FR_NUM_TENS:
            group += _FR_NUM_TENS[w]
            i += 1
            continue
        if w in _FR_NUM_MULT:
            base = group if group > 0 else 1
            group = base * _FR_NUM_MULT[w]
            i += 1
            continue
        if w in _FR_NUM_THOUSAND:
            base = group if group > 0 else 1
            total += base * _FR_NUM_THOUSAND[w]
            group = 0
            i += 1
            continue
        if w in _FR_NUM_SCALE:
            base = group if group > 0 else 1
            total += base * _FR_NUM_SCALE[w]
            group = 0
            i += 1
            continue
        return None  # sous-mot non reconnu dans ce contexte -> séquence invalide
    return total + group


def _convert_number_words(text: str) -> str:
    """FIX #8 : remplace les SÉQUENCES de mots-nombres FR par leur valeur composée
    (ex. 'trois mille cinq cents' -> '3500'), au lieu d'une substitution mot-à-mot
    qui ne composait rien (l'ancienne version v2 donnait 'trois mille cinq cents'
    -> '3 1000 5 100', sans aucune arithmétique)."""
    raw_tokens = text.split()
    out: list[str] = []
    i, n = 0, len(raw_tokens)
    while i < n:
        # Garde-fou : 'cent'/'cents' dans l'idiome 'pour cent' n'est pas un numéral isolé.
        if raw_tokens[i].lower() in ("cent", "cents") and i > 0 and raw_tokens[i - 1].lower() == "pour":
            out.append(raw_tokens[i])
            i += 1
            continue
        j = i
        subwords: list[str] = []
        while j < n:
            parts = raw_tokens[j].lower().split("-")
            if all(p in _FR_NUM_VOCAB or p == "et" for p in parts):
                subwords.extend(parts)
                j += 1
                if (j < n and raw_tokens[j].lower() == "et"
                        and j + 1 < n and raw_tokens[j + 1].lower() in ("un", "une", "onze")):
                    subwords.append("et")
                    subwords.append(raw_tokens[j + 1].lower())
                    j += 2
            else:
                break
        if j > i and subwords:
            value = _parse_number_run(subwords)
            if value is not None:
                out.append(str(value))
                i = j
                continue
        out.append(raw_tokens[i])
        i += 1
    return " ".join(out)


def _currency_repl(m: re.Match) -> str:
    # FIX #4 : préserve la devise réelle (eur/usd/gbp) au lieu de tout figer en "eur",
    # ce qui aurait masqué une vraie erreur de devise entre référence et hypothèse.
    return f"{m.group(1)} {_CURRENCY_MAP[m.group(2).lower()]}"


def normalise_numbers(text: str) -> str:
    """
    Normalise les variantes de format de nombres pour ne pas pénaliser injustement
    un moteur qui écrit "3 500 euros" alors que la référence dit
    "trois mille cinq cents euros" — les deux convergent vers "3500 eur".

    Ordre des opérations (FIX #8 : la composition des nombres en lettres passe
    désormais EN PREMIER, pour que les étapes suivantes voient des chiffres) :
    0. Composer les nombres écrits en lettres FR (vocabulaire courant).
    1. Fusionner les milliers séparés par espace : "3 500" -> "3500".
    2. Normaliser les devises (symboles ET abréviations) en préservant la devise réelle.
    3. Normaliser les pourcentages : "12 %" / "12 pourcent" -> "12 pct".
    4. Normaliser les dates : "01/04/2024" -> "01 04 2024".
    """
    text = _convert_number_words(text)

    for _ in range(4):  # itère pour les nombres à plusieurs groupes ("1 234 567")
        text = _FUSED_THOUSANDS.sub(r"\1\2", text)

    text = _CURRENCY_SYMBOLS.sub(_currency_repl, text)
    text = _PERCENT.sub(lambda m: m.group(1) + " pct", text)
    text = _DATE_SEP.sub(lambda m: f"{m.group(1)} {m.group(2)} {m.group(3)}", text)
    return text


# ──────────────────────────────────────────────────────────────────────────────
# Edit-distance core (Wagner-Fischer DP)
# ──────────────────────────────────────────────────────────────────────────────

def _edit_distance_matrix(ref: list[str], hyp: list[str]) -> list[list[int]]:
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d


def _backtrace(d: list[list[int]], ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    i, j = len(ref), len(hyp)
    subs = dels = ins = 0
    while i > 0 or j > 0:
        if i == 0:
            ins += 1
            j -= 1
        elif j == 0:
            dels += 1
            i -= 1
        else:
            diag, up, left = d[i - 1][j - 1], d[i - 1][j], d[i][j - 1]
            best = min(diag, up, left)
            if best == diag:
                if ref[i - 1] != hyp[j - 1]:
                    subs += 1
                i -= 1
                j -= 1
            elif best == up:
                dels += 1
                i -= 1
            else:
                ins += 1
                j -= 1
    return subs, dels, ins


def _edit_distance_no_backtrace(ref: list[str], hyp: list[str]) -> int:
    """Distance d'édition O(min(N,M)) en mémoire (pas de backtrace) — utilisée par la CER."""
    if len(ref) < len(hyp):
        ref, hyp = hyp, ref
    n, m = len(ref), len(hyp)
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m]


# ──────────────────────────────────────────────────────────────────────────────
# Standard Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_wer(reference: str, hypothesis: str) -> dict[str, float | int]:
    """Word Error Rate with full edit-distance breakdown."""
    ref_tokens = tokenise(reference)
    hyp_tokens = tokenise(hypothesis)
    n = len(ref_tokens)
    if n == 0:
        hyp_len = len(hyp_tokens)
        if hyp_len == 0:
            return {
                "wer": 0.0, "word_accuracy": 1.0,
                "substitutions": 0, "deletions": 0, "insertions": 0,
                "ref_length": 0, "hyp_length": 0,
            }
        # Référence vide, hypothèse non vide : indéfini mathématiquement (division par
        # zéro) ; renvoie inf pour ne jamais masquer une hallucination sur silence.
        # -> Utiliser to_json_safe() avant toute sérialisation JSON stricte (FIX #10),
        #    et aggregate_wer_corpus() pour l'agrégation multi-énoncés (FIX #10).
        return {
            "wer": float("inf"), "word_accuracy": 0.0,
            "substitutions": 0, "deletions": 0, "insertions": hyp_len,
            "ref_length": 0, "hyp_length": hyp_len,
        }

    d = _edit_distance_matrix(ref_tokens, hyp_tokens)
    subs, dels, ins = _backtrace(d, ref_tokens, hyp_tokens)
    wer = (subs + dels + ins) / n
    return {
        "wer": round(wer, 6),
        "word_accuracy": round(max(0.0, 1.0 - wer), 6),
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "ref_length": n,
        "hyp_length": len(hyp_tokens),
    }


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate (spaces removed before comparison)."""
    ref_chars = list(normalise(reference).replace(" ", ""))
    hyp_chars = list(normalise(hypothesis).replace(" ", ""))
    if not ref_chars:
        # FIX #2 : cohérent avec compute_wer — ne masque plus une hallucination
        # sur silence derrière un score parfait de 0.0.
        return 0.0 if not hyp_chars else float("inf")
    dist = _edit_distance_no_backtrace(ref_chars, hyp_chars)
    return round(dist / len(ref_chars), 6)


def compute_rtf(latency_sec: float, audio_duration_sec: float) -> float:
    """Real-Time Factor = latency / audio_duration."""
    return round(latency_sec / audio_duration_sec, 4) if audio_duration_sec > 0 else 0.0


def estimate_cost(audio_duration_sec: float, cost_per_min: float) -> float:
    """Estimate transcription cost based on $/min rate."""
    return round((audio_duration_sec / 60.0) * cost_per_min, 6) if cost_per_min > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Extended Standard Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_mer(reference: str, hypothesis: str) -> float:
    """Match Error Rate — (S+D+I)/(N+I), conforme à Morris, Maier & Green (2004).
    FIX #5 : v2 utilisait /max(N,P), qui diverge de la définition standard dès que
    substitutions, suppressions ET insertions coexistent dans le même alignement."""
    ref_toks, hyp_toks = tokenise(reference), tokenise(hypothesis)
    n, p = len(ref_toks), len(hyp_toks)
    if n == 0 and p == 0:
        return 0.0
    d = _edit_distance_matrix(ref_toks, hyp_toks)
    subs, dels, ins = _backtrace(d, ref_toks, hyp_toks)
    denom = n + ins
    return round((subs + dels + ins) / denom, 6) if denom > 0 else 0.0


def compute_wip_wil(reference: str, hypothesis: str) -> dict[str, float]:
    """Word Information Preserved (WIP) et Lost (WIL) — WIP = (H/N)·(H/P)."""
    ref_toks, hyp_toks = tokenise(reference), tokenise(hypothesis)
    n, p = len(ref_toks), len(hyp_toks)
    if n == 0 and p == 0:
        # FIX #6 : cohérent avec compute_wer (wer=0.0/accuracy=1.0 sur ce même cas) —
        # silence parfaitement transcrit = succès, pas le pire score possible.
        return {"wip": 1.0, "wil": 0.0}
    if n == 0 or p == 0:
        return {"wip": 0.0, "wil": 1.0}
    d = _edit_distance_matrix(ref_toks, hyp_toks)
    subs, dels, ins = _backtrace(d, ref_toks, hyp_toks)
    hits_ref = max(0, n - subs - dels)
    hits_hyp = max(0, p - subs - ins)
    wip = round((hits_ref / n) * (hits_hyp / p), 6)
    return {"wip": wip, "wil": round(max(0.0, 1.0 - wip), 6)}


# ──────────────────────────────────────────────────────────────────────────────
# NEW v2 — Normalised WER
# ──────────────────────────────────────────────────────────────────────────────

def compute_norm_wer(reference: str, hypothesis: str) -> dict[str, float | int]:
    """WER calculée après normalisation des nombres (cf. normalise_numbers, FIX #4/#8)."""
    return compute_wer(normalise_numbers(reference), normalise_numbers(hypothesis))


# ──────────────────────────────────────────────────────────────────────────────
# Domain / Semantic Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_tner(reference: str, hypothesis: str, domain_lexicon: list[str]) -> float | None:
    """Term Miss Rate : proportion d'occurrences de termes du lexique présentes en
    référence mais absentes/sous-représentées en hypothèse.
    NB : aveugle aux termes HALLUCINÉS (en hypothèse, absents de la référence) —
    voir compute_term_metrics() (FIX #11) pour la vue précision complémentaire."""
    if not domain_lexicon:
        return None
    ref_toks = tokenise(reference)
    hyp_toks = tokenise(hypothesis)
    total_target = 0
    missed = 0
    for term in domain_lexicon:
        term_toks = tokenise(term)
        if not term_toks:
            continue
        tlen = len(term_toks)
        ref_count = sum(1 for i in range(len(ref_toks) - tlen + 1) if ref_toks[i:i + tlen] == term_toks)
        if ref_count > 0:
            total_target += ref_count
            hyp_count = sum(1 for i in range(len(hyp_toks) - tlen + 1) if hyp_toks[i:i + tlen] == term_toks)
            if hyp_count < ref_count:
                missed += ref_count - hyp_count
    return round(missed / total_target, 6) if total_target > 0 else None


def compute_term_metrics(reference: str, hypothesis: str, domain_lexicon: list[str]) -> dict | None:
    """FIX #11 : précision / rappel / F1 au niveau des termes métier — capte aussi
    les termes HALLUCINÉS (présents en hypothèse, absents de la référence), que
    compute_tner() ne voit pas par construction."""
    if not domain_lexicon:
        return None
    ref_toks, hyp_toks = tokenise(reference), tokenise(hypothesis)
    total_ref_occ = total_hyp_occ = total_matched = 0
    for term in domain_lexicon:
        term_toks = tokenise(term)
        if not term_toks:
            continue
        tlen = len(term_toks)
        ref_count = sum(1 for i in range(len(ref_toks) - tlen + 1) if ref_toks[i:i + tlen] == term_toks)
        hyp_count = sum(1 for i in range(len(hyp_toks) - tlen + 1) if hyp_toks[i:i + tlen] == term_toks)
        total_ref_occ += ref_count
        total_hyp_occ += hyp_count
        total_matched += min(ref_count, hyp_count)
    if total_ref_occ == 0 and total_hyp_occ == 0:
        return None
    precision = total_matched / total_hyp_occ if total_hyp_occ > 0 else 0.0
    recall = total_matched / total_ref_occ if total_ref_occ > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "term_precision": round(precision, 6),
        "term_recall": round(recall, 6),
        "term_f1": round(f1, 6),
        "ref_occurrences": total_ref_occ,
        "hyp_occurrences": total_hyp_occ,
    }


def compute_har(reference: str, hypothesis: str) -> float:
    """Over-Generation Rate (OGR) — parfois appelée 'Hallucination Rate' en ASR.
    NOTE : approximation basée sur la référence ; ne détecte pas les hallucinations
    acoustiques sur audio réellement silencieux/hors-vocabulaire (approche reference-free)."""
    ref_counts = Counter(tokenise(reference))
    hyp_toks = tokenise(hypothesis)
    if not hyp_toks:
        return 0.0
    hyp_counts = Counter(hyp_toks)
    hallucinated = sum(max(0, c - ref_counts[w]) for w, c in hyp_counts.items())
    return round(hallucinated / len(hyp_toks), 6)


# ──────────────────────────────────────────────────────────────────────────────
# Punctuation & Readability
# ──────────────────────────────────────────────────────────────────────────────

def compute_punctuation_metrics(reference: str, hypothesis: str) -> dict[str, float]:
    """Punctuation Macro F1 et Punctuation Placement Error Rate (PPER)."""
    PUNCT_CLASSES = {"COMMA": r",", "PERIOD": r"\.", "QUESTION": r"\?", "EXCLAM": r"!"}
    PUNCT_WEIGHTS = {"PERIOD": 1.0, "QUESTION": 1.0, "EXCLAM": 0.9, "COMMA": 0.6}

    def punct_class(word: str) -> str | None:
        # FIX #7 : ancré en fin de mot ($) — ne confond plus une virgule décimale
        # ("12,50") ou un point abréviatif interne avec une ponctuation de phrase.
        for cls, pattern in PUNCT_CLASSES.items():
            if re.search(pattern + r"$", word):
                return cls
        return None

    ref_words = reference.split()
    hyp_words = hypothesis.split()
    ref_stripped = [re.sub(r'[.,?!]+$', '', w) for w in ref_words]
    hyp_stripped = [re.sub(r'[.,?!]+$', '', w) for w in hyp_words]

    d = _edit_distance_matrix(ref_stripped, hyp_stripped)
    alignment: list[tuple[int | None, int | None]] = []
    i, j = len(ref_stripped), len(hyp_stripped)
    while i > 0 or j > 0:
        if i == 0:
            alignment.append((None, j - 1)); j -= 1
        elif j == 0:
            alignment.append((i - 1, None)); i -= 1
        else:
            diag, up, left = d[i - 1][j - 1], d[i - 1][j], d[i][j - 1]
            best = min(diag, up, left)
            if best == diag:
                alignment.append((i - 1, j - 1)); i -= 1; j -= 1
            elif best == up:
                alignment.append((i - 1, None)); i -= 1
            else:
                alignment.append((None, j - 1)); j -= 1
    alignment.reverse()

    tp = {c: 0 for c in PUNCT_CLASSES}
    fp = {c: 0 for c in PUNCT_CLASSES}
    fn = {c: 0 for c in PUNCT_CLASSES}
    for ri, hi in alignment:
        rc = punct_class(ref_words[ri]) if ri is not None else None
        hc = punct_class(hyp_words[hi]) if hi is not None else None
        if rc == hc and rc is not None:
            tp[rc] += 1
        else:
            if rc: fn[rc] += 1
            if hc: fp[hc] += 1

    f1s = []
    for cls in PUNCT_CLASSES:
        # FIX #3 : exclut les classes sans aucune observation (tp+fp+fn==0) de la
        # moyenne, au lieu de leur imposer F1=0.0 (ce qui plafonnait artificiellement
        # le score sur tout énoncé ne contenant pas les 4 classes simultanément).
        support = tp[cls] + fp[cls] + fn[cls]
        if support == 0:
            continue
        prec = tp[cls] / (tp[cls] + fp[cls]) if tp[cls] + fp[cls] > 0 else 0.0
        rec = tp[cls] / (tp[cls] + fn[cls]) if tp[cls] + fn[cls] > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 1.0  # rien à évaluer -> rien à pénaliser

    tot_err, tot_w = 0.0, 0.0
    for cls, w in PUNCT_WEIGHTS.items():
        ref_total = tp[cls] + fn[cls]
        if ref_total > 0:
            tot_err += w * fn[cls] / ref_total
            tot_w += w

    return {
        'macro_f1': round(macro_f1, 6),
        'pper': round(tot_err / tot_w if tot_w > 0 else 0.0, 6),
    }


def compute_onset_wer(reference: str, hypothesis: str, n_tokens: int = 3) -> float | None:
    """Computes WER strictly on the first N tokens to evaluate onset quality (e.g., VAD clipping)."""
    ref_toks = tokenise(reference)[:n_tokens]
    hyp_toks = tokenise(hypothesis)[:n_tokens]
    n = len(ref_toks)
    if n == 0:
        return None
    d = _edit_distance_matrix(ref_toks, hyp_toks)
    subs, dels, ins = _backtrace(d, ref_toks, hyp_toks)
    wer = (subs + dels + ins) / n
    return round(wer, 6)


# ──────────────────────────────────────────────────────────────────────────────
# Diarisation — FIX #1
# ──────────────────────────────────────────────────────────────────────────────

_DIAR_LINE_RE = re.compile(r"\[(\d{1,3}):(\d{2}\.\d+)\s*-\s*(\d{1,3}):(\d{2}\.\d+)\]\s*([^:]+):")


def _extract_diarization_lines(reference_text: str) -> tuple[list[tuple[float, float, str]], int, int]:
    """Parsing pur Python (sans dépendance pyannote) -> testable unitairement.
    Retourne (segments, total_lignes_non_vides, lignes_matchées).
    FIX #1 (partiel) : minutes sur 1 à 3 chiffres (au lieu de 2 fixes), pour
    supporter les enregistrements de plus de 99 minutes."""
    lines = [l for l in reference_text.splitlines() if l.strip()]
    segments: list[tuple[float, float, str]] = []
    for line in lines:
        m = _DIAR_LINE_RE.search(line)
        if m:
            m1, s1, m2, s2, speaker = m.groups()
            start_sec = int(m1) * 60 + float(s1)
            end_sec = int(m2) * 60 + float(s2)
            segments.append((start_sec, end_sec, speaker.strip()))
    return segments, len(lines), len(segments)


def compute_diarization_metrics(
    reference_text: str,
    hypothesis_segments: list["DiarizationSegment"],
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

    segments, total_lines, matched_lines = _extract_diarization_lines(reference_text)

    # FIX #1 : ne jamais renvoyer un score "parfait" silencieux sur échec de parsing.
    if total_lines > 0 and matched_lines == 0:
        raise ValueError(
            f"compute_diarization_metrics: 0/{total_lines} lignes de référence n'ont pu être "
            "parsées (format attendu '[MM:SS.s - MM:SS.s] Speaker:'). Calcul annulé pour "
            "éviter un faux score parfait."
        )
    if 0 < matched_lines < total_lines:
        warnings.warn(
            f"compute_diarization_metrics: seulement {matched_lines}/{total_lines} lignes "
            "de référence ont été parsées ; le DER/JER calculé est partiel.",
            stacklevel=2,
        )

    ref_annotation = Annotation()
    for start, end, speaker in segments:
        ref_annotation[Segment(start, end)] = speaker

    hyp_annotation = Annotation()
    for seg in hypothesis_segments:
        hyp_annotation[Segment(seg.start_sec, seg.end_sec)] = seg.speaker

    if not ref_annotation.labels():
        # Ici total_lines == 0 : référence légitimement vide (pas un échec de parsing).
        return {"der": 0.0, "jer": 0.0}

    der_metric = DiarizationErrorRate(collar=0.250)
    jer_metric = JaccardErrorRate()
    return {
        "der": round(der_metric(ref_annotation, hyp_annotation), 4),
        "jer": round(jer_metric(ref_annotation, hyp_annotation), 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Agrégation de corpus & sérialisation — FIX #10
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_wer_corpus(per_utterance_results: list[dict]) -> dict:
    """WER de corpus poolée : Σ erreurs / Σ longueur_référence — PAS une moyenne
    arithmétique des WER par énoncé, qui est biaisée vers les énoncés courts et
    se trouve corrompue par un seul wer=inf (référence vide) non filtré.
    Les énoncés à référence vide sont exclus du calcul et comptés explicitement."""
    total_errors = 0
    total_ref_len = 0
    n_excluded_empty_ref = 0
    for r in per_utterance_results:
        if r["ref_length"] == 0:
            n_excluded_empty_ref += 1
            continue
        total_errors += r["substitutions"] + r["deletions"] + r["insertions"]
        total_ref_len += r["ref_length"]
    pooled_wer = total_errors / total_ref_len if total_ref_len > 0 else None
    return {
        "pooled_wer": round(pooled_wer, 6) if pooled_wer is not None else None,
        "n_utterances": len(per_utterance_results),
        "n_excluded_empty_ref": n_excluded_empty_ref,
        "total_errors": total_errors,
        "total_ref_length": total_ref_len,
    }


def to_json_safe(value):
    """Remplace récursivement inf/-inf/NaN par None, pour produire un JSON
    strictement conforme RFC 8259 (consommable par tout parseur strict),
    au lieu du littéral non standard 'Infinity' que json.dumps() accepte
    par défaut en Python mais que beaucoup de consommateurs externes rejettent."""
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):  # value != value -> NaN
            return None
        return value
    if isinstance(value, dict):
        return {k: to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_json_safe(v) for v in value]
    return value
