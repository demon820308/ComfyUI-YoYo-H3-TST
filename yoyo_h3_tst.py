"""YoYo H3 Temporal State Transport (TST).

Training-free temporal-attention correction for MiniMax H3, implementing
the TST method of "Temporal State Transport in Video Generation"
(arXiv:2609.08505): diagnose each attention head's frame-level transport
operator with Spectral Tension, then apply a homeostatic per-head query
temperature that sharpens over-mixed heads and softens fragmented ones.

This is an original implementation written from the paper's published
formulas (row entropy vs. von Neumann entropy of the frame-level
operator, cosine layer/step schedules). The H3 integration hooks the
attention through ComfyUI's ``optimized_attention_override`` transformer
option and learns the video layout (frames / rows per frame / sampling
step) from a diffusion-model forward wrapper. Any attention override
already installed on the model (e.g. Sol-Attn) is chained, not replaced:
the TST query scaling is applied first, then the previous override runs.

Only the video rows of the packed H3 sequence are touched; text,
keyframe/reference and audio rows pass through unchanged. If anything
about the call does not match the expected H3 layout, the call falls
through to the original attention untouched.
"""

import logging
import math

import torch

import comfy.patcher_extension

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Core math (kept free of ComfyUI state so it can be unit-tested standalone)
# ---------------------------------------------------------------------------

def spectral_tension(q_video, k_video, frames, rows_per_frame):
    """Per-head Spectral Tension of the frame-level transport operator.

    q_video / k_video: post-RoPE tensors of shape (heads, S_v, head_dim)
    covering only the video rows of the packed sequence, in frame order.

    Returns a float32 tensor of shape (heads,): T > 0 means over-mixed
    transport, T < 0 means fragmented transport (paper Eqs. 1-3).
    """
    heads, _, head_dim = q_video.shape
    qv = q_video.reshape(heads, frames, rows_per_frame, head_dim).mean(dim=2).float()
    kv = k_video.reshape(heads, frames, rows_per_frame, head_dim).mean(dim=2).float()
    a = torch.softmax(qv @ kv.transpose(-2, -1) * (head_dim ** -0.5), dim=-1).clamp(min=_EPS)

    log_f = math.log(frames)
    h_row = -(a * a.log()).sum(dim=-1).mean(dim=-1) / log_f

    gram = a @ a.transpose(-2, -1)
    trace = gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1).clamp(min=_EPS)
    eig = torch.linalg.eigvalsh(gram / trace[:, None, None]).clamp(min=_EPS)
    eig = eig / eig.sum(dim=-1, keepdim=True).clamp(min=_EPS)
    h_vn = -(eig * eig.log()).sum(dim=-1) / log_f

    return h_row - h_vn


def layer_weight(layer, total_layers):
    """Cosine schedule: deeper layers are corrected more strongly."""
    if total_layers <= 1:
        return 1.0
    return 0.5 - 0.5 * math.cos(math.pi * layer / (total_layers - 1))


def step_weight(step, total_steps):
    """Cosine schedule: earlier denoising steps are corrected more strongly."""
    if total_steps <= 1 or step is None:
        return 1.0
    return 0.5 + 0.5 * math.cos(math.pi * step / (total_steps - 1))


def head_gamma(tension, tau, layer, total_layers, step, total_steps):
    """Per-head query temperature gamma = exp(T * tau_eff)."""
    tau_eff = tau * layer_weight(layer, total_layers) * step_weight(step, total_steps)
    return torch.exp(tension * tau_eff)


# ---------------------------------------------------------------------------
# ComfyUI integration
# ---------------------------------------------------------------------------

def _find_video_tensor(x, depth=0):
    """Locate the 5D (B, C, T, H, W) video latent inside the packed input."""
    if depth > 3:
        return None
    if isinstance(x, torch.Tensor):
        return x if x.dim() == 5 else None
    if isinstance(x, (list, tuple)):
        for item in x:
            found = _find_video_tensor(item, depth + 1)
            if found is not None:
                return found
        return None
    unbind = getattr(x, "unbind", None)
    if callable(unbind):
        try:
            return _find_video_tensor(list(unbind()), depth + 1)
        except Exception:
            return None
    return None


def _find_transformer_options(args, kwargs):
    if isinstance(kwargs.get("transformer_options"), dict):
        return kwargs["transformer_options"]
    for a in args:
        if isinstance(a, dict) and ("sample_sigmas" in a or "transformer_options" in a):
            return a
    return {}


def _step_from(timestep, transformer_options):
    """Current (step, total_steps) from the running sigma schedule."""
    schedule = transformer_options.get("sample_sigmas")
    if schedule is None or timestep is None:
        return None, None
    try:
        sigma = float(timestep.flatten()[0]) / 1000.0
        sched = schedule.float().flatten()
        step = int((sched - sigma).abs().argmin())
        return step, max(1, sched.shape[0] - 1)
    except Exception:
        return None, None


def _make_forward_wrapper(state):
    def wrapper(executor, *args, **kwargs):
        state["active"] = False
        state["attn_calls"] = 0
        state["stat_calls"] = 0
        state["stat_tension"] = 0.0
        state["stat_gamma_dev"] = 0.0
        try:
            x = args[0] if args else kwargs.get("x")
            video = _find_video_tensor(x)
            if video is not None:
                frames = int(video.shape[2])
                rows = ((int(video.shape[3]) + 1) // 2) * ((int(video.shape[4]) + 1) // 2)
                timestep = args[1] if len(args) > 1 else kwargs.get("timestep")
                step, total_steps = _step_from(timestep, _find_transformer_options(args, kwargs))
                state.update(frames=frames, rows=rows, step=step, total_steps=total_steps, active=frames >= 2)
        except Exception:
            state["active"] = False
        try:
            return executor(*args, **kwargs)
        finally:
            if state["active"] and state["log"] and state["stat_calls"] > 0:
                n = state["stat_calls"]
                logging.info(
                    "[YoYo TST] frames=%d step=%s/%s heads corrected=%d mean|T|=%.4f mean|gamma-1|=%.4f",
                    state["frames"],
                    state["step"] if state["step"] is not None else "?",
                    state["total_steps"] if state["total_steps"] is not None else "?",
                    n,
                    state["stat_tension"] / n,
                    state["stat_gamma_dev"] / n,
                )
            state["active"] = False

    return wrapper


def _make_attention_override(state, chained):
    def override(func, q, k, v, heads, mask=None, **kwargs):
        try:
            _apply_tst(state, q, k)
        except Exception:
            pass  # TST must never break sampling; fall through unmodified.
        if chained is not None:
            return chained(func, q, k, v, heads, mask=mask, **kwargs)
        return func(q, k, v, heads, mask=mask, **kwargs)

    return override


def _apply_tst(state, q, k):
    if not state.get("active"):
        return
    if q.dim() != 4 or k.dim() != 4:
        return
    frames, rows = state["frames"], state["rows"]
    seq = q.shape[2]
    video_rows = frames * rows
    if frames < 2 or seq <= video_rows:
        return  # no video segment in this call (e.g. text refiner): skip

    layer = state["attn_calls"]
    state["attn_calls"] = layer + 1
    layer_idx = layer % state["num_layers"] if state["num_layers"] > 0 else 0

    q_video = q[0, :, seq - video_rows:]
    k_video = k[0, :, seq - video_rows:]
    tension = spectral_tension(q_video, k_video, frames, rows)

    if state["tau"] != 0.0:
        gamma = head_gamma(tension, state["tau"], layer_idx, state["num_layers"], state["step"], state["total_steps"])
        q[:, :, seq - video_rows:] = q[:, :, seq - video_rows:] * gamma.to(q.dtype)[None, :, None, None]
        state["stat_gamma_dev"] += float((gamma - 1.0).abs().mean())
    state["stat_tension"] += float(tension.abs().mean())
    state["stat_calls"] += 1


def patch_model(model, tau, log_diagnostics):
    diffusion_model = model.get_model_object("diffusion_model")
    mro_names = [c.__name__ for c in type(diffusion_model).__mro__]
    if not any("MiniMaxH3" in name for name in mro_names):
        logging.warning("YoYo_H3TemporalStateTransport: model is not MiniMax H3; returning it unchanged.")
        return model

    blocks = getattr(diffusion_model, "blocks", None)
    state = {
        "tau": float(tau),
        "log": bool(log_diagnostics),
        "num_layers": len(blocks) if blocks else 0,
        "active": False,
        "frames": 0,
        "rows": 0,
        "step": None,
        "total_steps": None,
        "attn_calls": 0,
        "stat_calls": 0,
        "stat_tension": 0.0,
        "stat_gamma_dev": 0.0,
    }

    patched = model.clone()
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        "yoyo_h3_tst",
        _make_forward_wrapper(state),
    )
    transformer_options = patched.model_options.setdefault("transformer_options", {})
    chained = transformer_options.get("optimized_attention_override")
    transformer_options["optimized_attention_override"] = _make_attention_override(state, chained)
    if chained is not None:
        logging.info("YoYo_H3TemporalStateTransport: chaining after an existing attention override.")
    return patched


class YoYo_H3TemporalStateTransport:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "tau": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Homeostatic correction strength; 0.2 is the paper setting. 0 disables correction while keeping diagnostics active.",
                }),
                "log_diagnostics": ("BOOLEAN", {
                    "default": True, "label_on": "诊断日志：开启", "label_off": "诊断日志：关闭",
                    "tooltip": "Log one [YoYo TST] line per model forward with mean tension and correction statistics.",
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "YoYo/MiniMax H3"
    DESCRIPTION = (
        "Temporal State Transport (TST) for MiniMax H3: per-head spectral-tension diagnosis of the "
        "frame-level attention operator, corrected by a scheduled query temperature. Training-free, "
        "works with any sampler, chains with existing attention overrides (e.g. Sol-Attn)."
    )

    def patch(self, model, tau, log_diagnostics):
        if tau == 0 and not log_diagnostics:
            return (model,)
        return (patch_model(model, tau, log_diagnostics),)


NODE_CLASS_MAPPINGS = {
    "YoYo_H3TemporalStateTransport": YoYo_H3TemporalStateTransport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YoYo_H3TemporalStateTransport": "YoYo H3 Temporal State Transport (TST)",
}
