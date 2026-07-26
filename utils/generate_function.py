import torch
from dynamic_dllm_cache.cache import DynamicDLLMCache
import torch.nn.functional as F
import numpy as np


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits.exp()
    noise = torch.rand_like(logits)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def extend_attention_mask(attention_mask, total_length):
    """Pad a prompt-length attention mask out to the full denoising sequence.

    The model is called on ``[prompt | gen_region]``, but the tokenizer only
    produces a mask for the prompt. Handing that short mask to the model builds
    an attention bias whose key dimension is ``prompt_length`` while q/k are
    ``prompt_length + gen_length``, which fails to broadcast inside
    scaled_dot_product_attention. Generated positions are always attendable, so
    the mask is extended with ones.
    """
    if attention_mask is None:
        return None
    batch_size, mask_length = attention_mask.shape
    if mask_length == total_length:
        return attention_mask
    if mask_length > total_length:
        raise ValueError(
            f"attention_mask is longer ({mask_length}) than the sequence ({total_length})"
        )
    pad = torch.ones(
        (batch_size, total_length - mask_length),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    return torch.cat([attention_mask, pad], dim=1)


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = base.expand(-1, steps).clone()
    if remainder.sum() > 0:
        indices = torch.arange(steps, device=mask_index.device)
        mask = indices.unsqueeze(0) < remainder
        num_transfer_tokens[mask] += 1
    return num_transfer_tokens.to(torch.int64)


def generate(
    input_ids,
    attention_mask,
    model,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
):
    """Fixed-schedule block-wise masked diffusion generation.

    Returns:
        (generation, nfe) where ``generation`` is the generated region only,
        shape ``(B, gen_length)``, and ``nfe`` is the number of denoising steps
        taken (one model forward per step when ``cfg_scale == 0``).
    """
    with torch.no_grad():
        batch_size, prompt_length = input_ids.shape
        x = torch.full(
            (batch_size, prompt_length + gen_length),
            mask_id,
            dtype=torch.long,
            device=model.device,
        )
        x[:, :prompt_length] = input_ids
        attention_mask = extend_attention_mask(attention_mask, x.shape[1])

        prompt_index = x != mask_id

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks

        feature_cache = DynamicDLLMCache()
        feature_cache.reset_cache(prompt_length)
        feature_cache.set_parameter(
            (feature_cache.select_from, feature_cache.window_size, feature_cache.layer_budget)
        )
        feature_cache.set_cur_prompt(torch.arange(prompt_length, device=model.device))
        nfe = 0
        for num_block in range(num_blocks):
            start_idx = prompt_length + num_block * block_length
            end_idx = prompt_length + (num_block + 1) * block_length

            block_x = x[:, start_idx:end_idx]
            block_mask_index = block_x == mask_id
            num_transfer_tokens = get_num_transfer_tokens(
                block_mask_index, steps_per_block
            )

            for i in range(steps_per_block):
                nfe += 1
                mask_index = x == mask_id
                if cfg_scale > 0.0:
                    if hasattr(feature_cache, "cfg_interval_steps"):
                        feature_cache.update_step(layer_id=33)
                        if feature_cache.refresh_cfg(layer_id=33):
                            cfg_x = x.clone()
                            cfg_x[prompt_index] = mask_id
                            logits = model(x, attention_mask=attention_mask).logits[
                                :, prompt_length:
                            ]
                            feature_cache.cache_type = "cfg"
                            cfg_logits = model(
                                cfg_x, attention_mask=attention_mask
                            ).logits[:, prompt_length:]
                            cfg_residual = logits - cfg_logits
                            feature_cache.set_cache(
                                layer_id=33,
                                feature_name="cfg_residual",
                                features=cfg_residual,
                                cache_type="gen",
                            )
                            feature_cache.cache_type = "no_cfg"
                        else:
                            feature_cache.cache_type = "cfg"
                            cfg_residual = feature_cache.get_cache(
                                layer_id=33,
                                feature_name="cfg_residual",
                                cache_type="gen",
                            )
                            feature_cache.cache_type = "no_cfg"
                            logits = model(x, attention_mask=attention_mask).logits[
                                :, prompt_length:
                            ]
                    else:
                        cfg_x = x.clone()
                        cfg_x[prompt_index] = mask_id
                        logits = model(x, attention_mask=attention_mask).logits[
                            :, prompt_length:
                        ]
                        cfg_logits = model(cfg_x, attention_mask=attention_mask).logits[
                            :, prompt_length:
                        ]
                        cfg_residual = logits - cfg_logits
                    logits = (logits - cfg_residual) + (cfg_scale + 1) * cfg_residual
                else:
                    logits = model(x, attention_mask=attention_mask).logits[
                        :, prompt_length:
                    ]
                if temperature == 0:
                    # add_gumbel_noise returns logits.exp(); exp is monotonic so
                    # the argmax is identical without the full-vocab copy.
                    x0 = torch.argmax(logits, dim=-1)
                else:
                    logits_with_noise = add_gumbel_noise(
                        logits, temperature=temperature
                    )
                    x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )
                elif remasking == "random":
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, (num_block + 1) * block_length :] = -np.inf

                x0 = torch.where(
                    mask_index[:, prompt_length:], x0, x[:, prompt_length:]
                )
                confidence = torch.where(mask_index[:, prompt_length:], x0_p, -np.inf)

                transfer_index = torch.zeros_like(
                    x0, dtype=torch.bool, device=x0.device
                )
                for j in range(confidence.shape[0]):
                    select_index = torch.topk(
                        confidence[j], k=num_transfer_tokens[j, i]
                    ).indices
                    transfer_index[j, select_index] = True
                x[:, prompt_length:][transfer_index] = x0[transfer_index]

                # Update cur_prompt for sliding window
                selected_positions = torch.where(transfer_index.any(dim=0))[0] + prompt_length
                if selected_positions.numel() > 0:
                    feature_cache.set_cur_prompt(selected_positions)
        return x[:, prompt_length:], nfe


# ===========================================================================
# Prediction Dynamics (PD) — adaptive threshold generation
# ===========================================================================

# The PD statistics are computed over the full vocabulary, so a whole
# (B, L, V) tensor exists per denoising step. For LLaDA that is
# B * L * 126464 elements, which at batch 16 / gen_length 512 is 7.7 GiB in
# float64 -- and the intermediates used to be several multiples of that. The
# helpers below chunk along the sequence dimension so peak memory stays close
# to the one tensor that genuinely has to exist, without changing any result.
PD_CHUNK = 64


def _softmax_chunked(logits, dtype, chunk=PD_CHUNK):
    """Softmax over the vocab, writing into one preallocated output tensor.

    ``F.softmax(logits.to(dtype), dim=-1)`` transiently holds both the upcast
    copy and the result; this holds one full tensor plus a small chunk.
    """
    out = torch.empty(logits.shape, dtype=dtype, device=logits.device)
    for start in range(0, logits.shape[1], chunk):
        stop = start + chunk
        out[:, start:stop] = F.softmax(logits[:, start:stop].to(dtype), dim=-1)
    return out


def _second_largest(probs, chunk=PD_CHUNK):
    """Second-largest probability per position.

    Equivalent to ``torch.sort(probs, dim=-1).values[:, :, -2]`` but via topk:
    a full sort also allocates an int64 index tensor the size of the input
    (another 7.7 GiB at batch 16), and only the top two values are ever read.
    """
    out = torch.empty(probs.shape[:2], dtype=probs.dtype, device=probs.device)
    for start in range(0, probs.shape[1], chunk):
        stop = start + chunk
        out[:, start:stop] = torch.topk(probs[:, start:stop], 2, dim=-1).values[:, :, 1]
    return out


def _cosine_similarity_chunked(a, b, chunk=PD_CHUNK):
    """Cosine similarity along the vocab dimension, chunked over positions."""
    out = torch.empty(a.shape[:2], dtype=a.dtype, device=a.device)
    for start in range(0, a.shape[1], chunk):
        stop = start + chunk
        out[:, start:stop] = F.cosine_similarity(
            a[:, start:stop], b[:, start:stop], dim=-1
        )
    return out


def _gather_softmax_prob(logits, index, dtype, chunk=PD_CHUNK):
    """Softmax probability of ``index``, without materialising the softmax.

    ``softmax(l)[i] == exp(l[i] - logsumexp(l))``, so only a (B, L) result and
    one chunk of upcast logits need to exist -- instead of a full (B, L, V)
    probability tensor that is then thrown away after a single gather.
    """
    out = torch.empty(logits.shape[:2], dtype=dtype, device=logits.device)
    for start in range(0, logits.shape[1], chunk):
        stop = start + chunk
        chunk_logits = logits[:, start:stop].to(dtype)
        selected = chunk_logits.gather(
            -1, index[:, start:stop].unsqueeze(-1)
        ).squeeze(-1)
        out[:, start:stop] = (selected - torch.logsumexp(chunk_logits, dim=-1)).exp()
    return out


def update_pd_threshold(logits, prev_probs, current_threshold, mask_index,
                         pd_mode, alpha, beta, global_step_counter,
                         dtype=torch.float64):
    """
    Update PD (Prediction Dynamics) threshold based on model confidence.

    Args:
        logits: Current logits for the relevant region (B, L, V).
        prev_probs: Softmax probs from previous step (B, L, V), or None for first step.
        current_threshold: scalar (mode 1) or tensor [B,L] (mode 2).
        mask_index: Bool tensor (B, L), True for masked positions.
        pd_mode: 1 = global scalar, 2 = per-token tensor.
        alpha: Weight for peak-confidence term.
        beta: Weight for distribution-shift term.
        global_step_counter: 1-based step count.
        dtype: accumulation dtype for the probability tensors. float64 matches
            the original implementation; float32 halves the memory these
            statistics need.

    Returns:
        (updated_threshold, current_probs)
    """
    if pd_mode == 0:
        return current_threshold, None

    probabilities = _softmax_chunked(logits, dtype)

    if global_step_counter > 1 and prev_probs is not None:
        peak_confidence = 1 - _second_largest(probabilities)

        distribution_similarity = 1 - _cosine_similarity_chunked(
            probabilities, prev_probs
        )

        if pd_mode == 1:
            pc_mean = torch.mean(peak_confidence[mask_index]) if mask_index.any() else torch.tensor(0.0, device=logits.device)
            ds_mean = torch.mean(distribution_similarity[mask_index]) if mask_index.any() else torch.tensor(0.0, device=logits.device)
            updated_threshold = current_threshold - alpha * pc_mean + beta * ds_mean
        else:  # pd_mode == 2
            updated_threshold = current_threshold - alpha * peak_confidence + beta * distribution_similarity

        return updated_threshold, probabilities
    else:
        if pd_mode == 2:
            current_threshold = torch.full(
                size=logits.shape[:2], fill_value=float(current_threshold),
                device=logits.device, dtype=dtype
            )
        return current_threshold, probabilities


def get_transfer_index_pd(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    threshold,
    pd_mode: int,
    dtype=torch.float64,
):
    """
    PD-aware token selection. Computes confidence the same way as get_transfer_index,
    then selects tokens whose confidence >= the adaptive PD threshold.

    At least one token (max confidence) is always transferred per batch row.

    Args:
        logits / mask_index / x: the generation region only, shape (B, gen_length).
            Prompt positions are never masked, so restricting to the generation
            region is equivalent and avoids upcasting prompt logits that are
            then discarded.

    Returns:
        x0: (B, gen_length) long — proposed tokens
        transfer_index: (B, gen_length) bool — which positions to update this step
    """
    if temperature == 0:
        # add_gumbel_noise returns logits.exp() here, and exp is monotonic, so
        # the argmax is unchanged -- skip materialising a full (B, L, V) copy.
        x0 = torch.argmax(logits, dim=-1)
    else:
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == "low_confidence":
        x0_p = _gather_softmax_prob(logits, x0, dtype)
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=dtype)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)

    transfer_index = mask_index & (confidence >= threshold)

    max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)
    force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)
    transfer_index = transfer_index | force_mask
    transfer_index = transfer_index & mask_index

    return x0, transfer_index


@torch.no_grad()
def generate_pd(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                remasking='low_confidence', mask_id=126336, threshold=None,
                pd_mode=2, pd_threshold=0.9, alpha=0.001, beta=0.0008,
                use_cache=False, cfg_scale=0.0, attention_mask=None,
                pd_dtype=torch.float64):
    """
    Block-wise generation with Prediction Dynamics (PD) adaptive threshold.

    Compatible with window_budget cache mode and no-cache mode.
    When use_cache=True, the DynamicDLLMCache singleton must already be
    configured (via DynamicDLLMCache.new_instance) and the model hooks must be registered
    (via register_cache_LLaDA) before calling this function.

    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length for semi-autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        remasking: 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] (default 126336).
        threshold: Fixed threshold for non-PD fallback (ignored when pd_mode > 0).
        pd_mode: 0=disabled, 1=global scalar threshold, 2=per-token threshold.
            The paper's Eq. 11 adapts a threshold per token, i.e. mode 2.
        pd_threshold: Initial threshold tau^T (paper Fig. 6c uses 0.9).
        alpha: PD update weight for peak confidence. The paper uses 0.001;
            it is sensitive to order-of-magnitude scaling (Table 7: alpha/beta
            of 0.01/0.008 drops GSM8K from 78.01 to 69.75, 0.1/0.08 to 59.76).
        beta: PD update weight for distribution shift. The paper uses 0.0008.
        use_cache: Enable dynamic dllm (default False).
        cfg_scale: Unsupervised classifier-free guidance scale (default 0.0).
        attention_mask: Attention mask for the prompt; extended internally to
            cover the generation region.
        pd_dtype: accumulation dtype for the PD probability statistics.
            torch.float64 reproduces the original implementation; torch.float32
            halves the memory they need at a cost of ~1e-7 relative error.
    """
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    attention_mask = extend_attention_mask(attention_mask, x.shape[1])

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    prompt_len = prompt.shape[1]

    # --- Cache initialization ---
    if use_cache:
        feature_cache = DynamicDLLMCache()
        feature_cache.reset_cache(prompt_len)
        feature_cache.set_parameter(
            (feature_cache.select_from, feature_cache.window_size, feature_cache.layer_budget)
        )
        feature_cache.set_cur_prompt(torch.arange(prompt_len, device=model.device))

    # --- PD state ---
    prev_probs = None
    current_pd_threshold = pd_threshold
    global_step_counter = 0

    nfe = 0
    for num_block in range(num_blocks):
        block_start = prompt_len + num_block * block_length
        block_end = prompt_len + (num_block + 1) * block_length

        block_mask_index = (x[:, block_start:block_end] == mask_id)
        i = 0
        while True:
            nfe += 1
            global_step_counter += 1

            mask_index = (x == mask_id)
            logits = model(x, attention_mask=attention_mask).logits
            mask_index[:, block_end:] = 0

            # --- Compute PD on full generation-region logits ---
            gen_logits = logits[:, prompt_len:, :]
            gen_mask = mask_index[:, prompt_len:]

            # --- Update PD threshold ---
            updated_threshold, cur_probs_full = update_pd_threshold(
                gen_logits, prev_probs, current_pd_threshold,
                gen_mask, pd_mode=pd_mode, alpha=alpha, beta=beta,
                global_step_counter=global_step_counter, dtype=pd_dtype,
            )
            current_pd_threshold = updated_threshold
            if cur_probs_full is not None:
                prev_probs = cur_probs_full

            # --- Build threshold for get_transfer_index_pd ---
            # Selection runs on the generation region only. pd_mode 2 already
            # produces a (B, gen_length) threshold, so no full-sequence
            # threshold tensor (previously padded with inf over the prompt)
            # needs to be built.
            if pd_mode == 1:
                use_threshold = float(current_pd_threshold) if not isinstance(current_pd_threshold, (float, int)) else current_pd_threshold
            else:
                use_threshold = current_pd_threshold

            x0, transfer_index = get_transfer_index_pd(
                gen_logits, temperature, remasking, gen_mask, x[:, prompt_len:],
                threshold=use_threshold, pd_mode=pd_mode, dtype=pd_dtype,
            )
            x[:, prompt_len:][transfer_index] = x0[transfer_index]

            # --- Update sliding window center ---
            if use_cache:
                selected_positions = torch.where(transfer_index.any(dim=0))[0] + prompt_len
                if selected_positions.numel() > 0:
                    feature_cache.set_cur_prompt(selected_positions)

            i += 1

            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break

    return x, nfe
