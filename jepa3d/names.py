"""Object names in dino.txt's text space: for the question side and as a second answer target.

Both need a model whose X-encoder is jepa3d.x_encoder_dinotxt.DinoTxtXEncoder
(its frozen text tower gives the vectors, the half that is aligned with patches).

Question side (config "slots: {enabled: true, mode: text}"): the vector of each
object name of the question goes through the same projections as the patch
features (the encoder's feat_proj, the model's vis_proj) and is added on the
name's tokens, so a name and the patches that show it start out similar.

Second target (config "name_target: {enabled: true}"): for questions whose
answer is one of several object names (name_options), a head on the pooled
predictor output must pick the answer's dino.txt text vector among the vectors
of all options of the batch. The main target (the answer sentence in the
y-encoder's space) is unchanged.
"""

import torch
import torch.nn.functional as F

TEMPERATURE = 0.07
NAME_TYPES = ("relative_distance_object", "relative_size_object")   # VSI-590K rows: "options", "target_short"
BENCH_TYPES = ("object_rel_distance",)                                # VSI-Bench rows: "candidates", "target"


def name_options(row: dict):
    """(option names, index of the answer) if the row's answer is one of several object names, else None."""
    kind = row.get("question_type") or row.get("type")
    if kind in NAME_TYPES:
        options, answer = row.get("options"), row.get("target_short")
    elif kind in BENCH_TYPES:
        options, answer = row.get("candidates"), row.get("target")
    else:
        return None
    if not options or answer not in options or row.get("delexicalized"):
        return None
    return list(options), options.index(answer)


def name_terms(encoder, name_pred: torch.Tensor, rows) -> dict:
    """name_pred (B, D) of the name head, the batch's rows ->
    {"loss" (answer among all option names of the batch), "hits" (per row: right among its own options), "rows"}."""
    found = [(i, name_options(row)) for i, row in enumerate(rows)]
    found = [(i, o) for i, o in found if o is not None]
    if not found:
        return {"loss": name_pred.sum() * 0.0, "hits": [], "rows": []}
    names = sorted({n for _, (options, _) in found for n in options})
    table = F.normalize(encoder.text_vectors(names).to(name_pred.device), dim=-1)           # (M, D)
    index = {n: k for k, n in enumerate(names)}
    rows_i = torch.tensor([i for i, _ in found], device=name_pred.device)
    logits = F.normalize(name_pred.float()[rows_i], dim=-1) @ table.T / TEMPERATURE          # (R, M)
    labels = torch.tensor([index[options[a]] for _, (options, a) in found], device=name_pred.device)
    hits = []
    for r, (_, (options, a)) in enumerate(found):
        own = logits[r, [index[n] for n in options]]
        hits.append(bool(own.argmax() == a))
    return {"loss": F.cross_entropy(logits, labels), "hits": hits, "rows": [i for i, _ in found]}
