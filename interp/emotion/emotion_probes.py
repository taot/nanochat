"""
Minimal emotion-probe experiments for nanochat.

This script reproduces a small, practical subset of:
https://transformer-circuits.pub/2026/emotions/index.html

Example:
python -m interp.emotion.emotion_probes extract --source sft --train-per-emotion 40 --test-per-emotion 20
python -m interp.emotion.emotion_probes eval-probe
python -m interp.emotion.emotion_probes logit-lens
python -m interp.emotion.emotion_probes steer --emotion happy --strength 2.0
"""

import argparse
import json
import os
import random
from collections import defaultdict
from contextlib import contextmanager, nullcontext

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tabulate import tabulate

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_init


DEFAULT_EMOTIONS = ["happy", "sad", "angry", "calm", "afraid", "desperate", "proud", "loving"]
DEFAULT_OUT_DIR = os.path.join("out", "emotion_probes")
DATASET_NAME = "ryancodrai/emotion-probes"
STORIES_FILE = "expression/stories.parquet"


def parse_args():
    parser = argparse.ArgumentParser(description="Extract and evaluate emotion vectors on nanochat")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    def add_common_model_args(p):
        p.add_argument("--source", choices=["base", "sft", "rl"], default="sft")
        p.add_argument("--model-tag", type=str, default=None)
        p.add_argument("--step", type=int, default=None)
        p.add_argument("--device-type", choices=["cuda", "cpu", "mps"], default="")

    extract = subparsers.add_parser("extract", help="Extract emotion vectors from story activations")
    add_common_model_args(extract)
    extract.add_argument("--emotions", nargs="+", default=DEFAULT_EMOTIONS)
    extract.add_argument("--train-per-emotion", type=int, default=40)
    extract.add_argument("--test-per-emotion", type=int, default=20)
    extract.add_argument("--max-len", type=int, default=256)
    extract.add_argument("--skip-tokens", type=int, default=20)
    extract.add_argument("--layer", type=int, default=None, help="Default: floor(2/3 * n_layer)")
    extract.add_argument("--seed", type=int, default=42)
    extract.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)

    eval_probe = subparsers.add_parser("eval-probe", help="Evaluate held-out story projections")
    add_common_model_args(eval_probe)
    eval_probe.add_argument("--vectors", type=str, default=os.path.join(DEFAULT_OUT_DIR, "vectors.pt"))
    eval_probe.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    eval_probe.add_argument("--center-act", action="store_true",
                            help="Subtract global_mean from held-out activations before cosine scoring")

    logit_lens = subparsers.add_parser("logit-lens", help="Print top and bottom unembed tokens per emotion vector")
    add_common_model_args(logit_lens)
    logit_lens.add_argument("--vectors", type=str, default=os.path.join(DEFAULT_OUT_DIR, "vectors.pt"))
    logit_lens.add_argument("--top-k", type=int, default=10)

    steer = subparsers.add_parser("steer", help="Measure next-token logprob changes under activation steering")
    add_common_model_args(steer)
    steer.add_argument("--vectors", type=str, default=os.path.join(DEFAULT_OUT_DIR, "vectors.pt"))
    steer.add_argument("--emotion", nargs="+", default=["happy"])
    steer.add_argument("--strength", type=float, default=2.0)
    steer.add_argument("--prompt", type=str, default="How does he feel?")
    steer.add_argument("--assistant-prefix", type=str, default="He feels")
    steer.add_argument("--targets", nargs="+", default=DEFAULT_EMOTIONS)
    steer.add_argument("--positions", choices=["all", "last"], default="all")
    steer.add_argument("--top-k", type=int, default=10)
    steer.add_argument("--gen-steps", type=int, default=20,
                       help="Number of tokens to generate under steering (0 = skip)")

    return parser.parse_args()


def init_model(args):
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, _, _, _, device = compute_init(device_type)
    model, tokenizer, _ = load_model(
        args.source,
        device,
        phase="eval",
        model_tag=args.model_tag,
        step=args.step,
    )
    return model, tokenizer, device


def load_story_dataset():
    return load_dataset(DATASET_NAME, data_files=STORIES_FILE, split="train")


def sample_stories(emotions, train_per_emotion, test_per_emotion, seed):
    ds = load_story_dataset()
    buckets = defaultdict(list)
    wanted = set(emotions)
    for row in ds:
        emotion = row["emotion"]
        if emotion in wanted:
            buckets[emotion].append(row["story"])

    rng = random.Random(seed)
    train, test = [], []
    missing = []
    for emotion in emotions:
        stories = buckets.get(emotion, [])
        needed = train_per_emotion + test_per_emotion
        if len(stories) < needed:
            missing.append((emotion, len(stories), needed))
            continue
        rng.shuffle(stories)
        train.extend((emotion, story) for story in stories[:train_per_emotion])
        test.extend((emotion, story) for story in stories[train_per_emotion:needed])

    if missing:
        lines = [f"{emotion}: found {found}, need {needed}" for emotion, found, needed in missing]
        raise ValueError("Insufficient stories for requested emotions:\n" + "\n".join(lines))
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


@contextmanager
def capture_layer_activations(model, layer_idx):
    captured = []

    def hook(_module, _inputs, output):
        captured.append(output.detach().float().cpu())

    handle = model.transformer.h[layer_idx].register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


@contextmanager
def steer_layer(model, layer_idx, vector, strength, positions):
    vector = vector.to(model.get_device())

    def hook(_module, _inputs, output):
        delta = strength * vector.to(output.dtype).view(1, 1, -1)
        if positions == "last":
            output = output.clone()
            output[:, -1:, :] = output[:, -1:, :] + delta
            return output
        return output + delta

    handle = model.transformer.h[layer_idx].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.inference_mode()
def story_activation(model, tokenizer, text, layer_idx, max_len, skip_tokens):
    bos = tokenizer.get_bos_token_id()
    ids = tokenizer.encode(text, prepend=bos)[:max_len]
    if len(ids) < 2:
        raise ValueError("Need at least two tokens for nanochat forward")
    input_ids = torch.tensor([ids], dtype=torch.long, device=model.get_device())
    with capture_layer_activations(model, layer_idx) as captured:
        model(input_ids)
    acts = captured[-1][0]
    start = min(skip_tokens, max(0, acts.size(0) - 1))
    return acts[start:].mean(dim=0)


def extract(args):
    model, tokenizer, _ = init_model(args)
    layer_idx = args.layer if args.layer is not None else int(model.config.n_layer * 2 / 3)
    train, test = sample_stories(args.emotions, args.train_per_emotion, args.test_per_emotion, args.seed)

    print(f"Using layer {layer_idx}/{model.config.n_layer - 1}")
    # TODO 试验使用其他层。也可以把所有层的数据都提取出来
    print(f"Extracting {len(train)} train and {len(test)} test activations")

    by_emotion = defaultdict(list)
    for i, (emotion, story) in enumerate(train, 1):
        by_emotion[emotion].append(story_activation(model, tokenizer, story, layer_idx, args.max_len, args.skip_tokens))
        if i % 25 == 0 or i == len(train):
            print(f"train activations: {i}/{len(train)}")

    all_train = torch.stack([act for acts in by_emotion.values() for act in acts])
    global_mean = all_train.mean(dim=0)
    vectors = {}
    means = {}
    for emotion in args.emotions:
        mean = torch.stack(by_emotion[emotion]).mean(dim=0)
        means[emotion] = mean
        vectors[emotion] = mean - global_mean

    test_acts = []
    for i, (emotion, story) in enumerate(test, 1):
        test_acts.append((emotion, story_activation(model, tokenizer, story, layer_idx, args.max_len, args.skip_tokens)))
        if i % 25 == 0 or i == len(test):
            print(f"test activations: {i}/{len(test)}")

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "vectors.pt")
    torch.save(
        {
            "vectors": vectors,
            "means": means,
            "global_mean": global_mean,
            "test_activations": test_acts,
            "emotions": args.emotions,
            "layer": layer_idx,
            "source": args.source,
            "train_per_emotion": args.train_per_emotion,
            "test_per_emotion": args.test_per_emotion,
            "max_len": args.max_len,
            "skip_tokens": args.skip_tokens,
        },
        path,
    )
    print(f"Saved {path}")


def load_vectors(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def normalized_dot(a, b):
    return F.cosine_similarity(a.float(), b.float(), dim=0).item()


def eval_probe(args):
    data = load_vectors(args.vectors)
    emotions = data["emotions"]
    vectors = data["vectors"]
    test_acts = data["test_activations"]
    global_mean = data["global_mean"]

    matrix = {emotion: {probe: [] for probe in emotions} for emotion in emotions}
    correct = 0
    total = 0
    for emotion, act in test_acts:
        a = (act - global_mean) if args.center_act else act
        scores = {probe: normalized_dot(a, vectors[probe]) for probe in emotions}
        for probe, score in scores.items():
            matrix[emotion][probe].append(score)
        pred = max(scores, key=scores.get)
        correct += int(pred == emotion)
        total += 1

    rows = []
    for emotion in emotions:
        row = [emotion]
        for probe in emotions:
            vals = matrix[emotion][probe]
            row.append(sum(vals) / len(vals))
        rows.append(row)

    headers = ["true\\probe"] + emotions
    print(tabulate(rows, headers=headers, floatfmt=".3f"))
    print(f"top-1 accuracy: {correct}/{total} = {correct / max(total, 1):.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "probe_matrix.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"headers": headers, "rows": rows, "accuracy": correct / max(total, 1)}, f, indent=2)
    print(f"Saved {out_path}")


def token_label(tokenizer, token_id):
    text = tokenizer.decode([int(token_id)])
    return text.replace("\n", "\\n")


@torch.inference_mode()
def logit_lens(args):
    model, tokenizer, _ = init_model(args)
    data = load_vectors(args.vectors)
    for emotion in data["emotions"]:
        vector = data["vectors"][emotion].to(model.get_device()).to(model.lm_head.weight.dtype)
        logits = model.lm_head(vector).float()[:model.config.vocab_size]
        top_vals, top_ids = torch.topk(logits, args.top_k)
        bot_vals, bot_ids = torch.topk(-logits, args.top_k)
        top = [f"{token_label(tokenizer, i)} ({v.item():.3f})" for v, i in zip(top_vals, top_ids)]
        bottom = [f"{token_label(tokenizer, i)} ({-v.item():.3f})" for v, i in zip(bot_vals, bot_ids)]
        print(f"\n{emotion}")
        print("  top:    " + ", ".join(top))
        print("  bottom: " + ", ".join(bottom))


def render_chat_prompt(tokenizer, prompt, assistant_prefix=""):
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    ids = [bos, user_start]
    ids.extend(tokenizer.encode(prompt))
    ids.extend([user_end, assistant_start])
    if assistant_prefix:
        ids.extend(tokenizer.encode(assistant_prefix))
    return ids


@torch.inference_mode()
def next_token_logprobs(model, tokenizer, prompt_ids, targets, layer_idx=None, vector=None, strength=0.0, positions="all"):
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.get_device())
    ctx = steer_layer(model, layer_idx, vector, strength, positions) if vector is not None else nullcontext()
    with ctx:
        logits = model(input_ids)[:, -1, :]
    logprobs = F.log_softmax(logits.float(), dim=-1)[0]
    rows = []
    for target in targets:
        ids = tokenizer.encode(" " + target)        # TODO: 这里试验不加空格
        if not ids:
            continue
        token_id = ids[0]
        rows.append((target, token_label(tokenizer, token_id), logprobs[token_id].item()))
    return rows


@torch.inference_mode()
def top_next_tokens(model, tokenizer, prompt_ids, k, layer_idx=None, vector=None, strength=0.0, positions="all"):
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.get_device())
    ctx = steer_layer(model, layer_idx, vector, strength, positions) if vector is not None else nullcontext()
    with ctx:
        logits = model(input_ids)[:, -1, :]
    logprobs = F.log_softmax(logits.float(), dim=-1)[0]
    vals, ids = torch.topk(logprobs, k)
    return [(token_label(tokenizer, int(i)), v.item()) for v, i in zip(vals, ids)]


@torch.inference_mode()
def generate_steered(model, tokenizer, prompt_ids, gen_steps, layer_idx=None, vector=None, strength=0.0, positions="all"):
    ctx = steer_layer(model, layer_idx, vector, strength, positions) if vector is not None else nullcontext()
    generated = []
    with ctx:
        for tok_id in model.generate(prompt_ids, max_tokens=gen_steps):
            generated.append(tok_id)
    return tokenizer.decode(generated)


def steer(args):
    model, tokenizer, _ = init_model(args)
    data = load_vectors(args.vectors)
    unknown = [e for e in args.emotion if e not in data["vectors"]]
    if unknown:
        raise ValueError(f"Unknown emotion(s) {unknown!r}; available: {', '.join(data['emotions'])}")
    layer_idx = data["layer"]
    prompt_ids = render_chat_prompt(tokenizer, args.prompt, args.assistant_prefix)

    print(f"prompt: {args.prompt!r}")
    print(f"assistant_prefix: {args.assistant_prefix!r}")
    print(f"layer={layer_idx}, strength={args.strength}, positions={args.positions}")

    baseline = next_token_logprobs(model, tokenizer, prompt_ids, args.targets)
    by_target = {target: (tok, lp) for target, tok, lp in baseline}
    base_top = top_next_tokens(model, tokenizer, prompt_ids, args.top_k)

    print("\ntop next tokens (baseline):")
    print(tabulate([[f"{tok} ({lp:.3f})"] for tok, lp in base_top], headers=["baseline"]))
    if args.gen_steps > 0:
        base_text = generate_steered(model, tokenizer, prompt_ids, args.gen_steps)
        print(f"\nbaseline generation:\n  {base_text!r}")

    for emotion in args.emotion:
        vector = data["vectors"][emotion]
        steered = next_token_logprobs(
            model,
            tokenizer,
            prompt_ids,
            args.targets,
            layer_idx=layer_idx,
            vector=vector,
            strength=args.strength,
            positions=args.positions,
        )
        rows = []
        for target, tok, lp in steered:
            _, base_lp = by_target[target]
            rows.append([target, tok, base_lp, lp, lp - base_lp])
        steered_top = top_next_tokens(
            model, tokenizer, prompt_ids, args.top_k,
            layer_idx=layer_idx, vector=vector, strength=args.strength, positions=args.positions,
        )

        print(f"\n=== emotion: {emotion} ===")
        print("top next tokens (steered):")
        print(tabulate([[f"{tok} ({lp:.3f})"] for tok, lp in steered_top], headers=["steered"]))
        print()
        print(tabulate(rows, headers=["target", "first token", "baseline", "steered", "delta"], floatfmt=".3f"))
        if args.gen_steps > 0:
            text = generate_steered(
                model, tokenizer, prompt_ids, args.gen_steps,
                layer_idx=layer_idx, vector=vector, strength=args.strength, positions=args.positions,
            )
            print(f"\nsteered generation ({emotion}):\n  {text!r}")




def main():
    args = parse_args()
    if args.cmd == "extract":
        extract(args)
    elif args.cmd == "eval-probe":
        eval_probe(args)
    elif args.cmd == "logit-lens":
        logit_lens(args)
    elif args.cmd == "steer":
        steer(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
