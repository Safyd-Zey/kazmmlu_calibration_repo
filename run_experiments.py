"""
Calibration and selective answering of LLMs on KazMMLU (Kazakh vs Russian).

  python run_experiments.py infer --model <key>     # GPU, once per model (see MODELS)
  python run_experiments.py analyze                 # CPU, writes results/results.json, tables, figures

The KazMMLU schema used below was checked on the dataset card (MBZUAI/KazMMLU, CC BY-NC 4.0):
37 subsets "<Subject> (<Level> in kaz|rus)", each with dev.csv (few-shot, excluded) and test.csv;
columns ID, Source, Group, Subject, Level, Question, Option A..E, Answer Key (A-E), Subset, Split, Language.
"""
import argparse, glob, json, os, re, time
import numpy as np

DATASET_ID = "MBZUAI/KazMMLU"
MODELS = {  # key: (Hugging Face id, display name, adapted to Kazakh?)
    "llama31": ("meta-llama/Llama-3.1-8B-Instruct", "Llama 3.1 8B Instruct", False),
    "qwen25": ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5 7B Instruct", False),
    "gemma2": ("google/gemma-2-9b-it", "Gemma 2 9B Instruct", False),
    "kazllm": ("issai/LLama-3.1-KazLLM-1.0-8B", "KazLLM 1.0 8B", True),
    "sherkala": ("inception42/Llama-3.1-Sherkala-8B-Chat", "Sherkala-Chat 8B", True),
}
LETTERS = ["A", "B", "C", "D", "E"]
INSTR = {
    "kk": "Төмендегі сұраққа жауап беріңіз. Тек дұрыс жауаптың әрпін (A, B, C, D немесе E) жазыңыз.",
    "ru": "Ответьте на вопрос. Укажите только букву правильного ответа (A, B, C, D или E).",
}
KZ_SPECIFIC = ["Kazakh History", "Kazakh Language", "Kazakh Literature"]
OUT = os.environ.get("RESULTS_DIR", "results")
SEEDS = list(range(20))
ALPHA, DELTA, N_BINS, N_BOOT = 0.10, 0.05, 15, 1000
TARGET_RISKS = [0.10, 0.20]


# ======================================================================= data
def load_items(local_dir=None):
    import pandas as pd
    if local_dir is None:
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(DATASET_ID, repo_type="dataset", allow_patterns=["*/test.csv"])
    frames = []
    for f in sorted(glob.glob(os.path.join(local_dir, "*", "test.csv"))):
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        df["Subset"] = os.path.basename(os.path.dirname(f))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    items = []
    for _, r in df.iterrows():
        opts = [r[f"Option {L}"].strip() for L in LETTERS if r.get(f"Option {L}", "").strip() != ""]
        ans = r["Answer Key"].strip().upper()[:1]
        if ans not in LETTERS[:len(opts)]:
            continue  # malformed row
        m = re.match(r"(.+?) \((.+) in (kaz|rus)\)", r["Subset"])
        items.append(dict(q=r["Question"], opts=opts, y=LETTERS.index(ans),
                          lang="kk" if m.group(3) == "kaz" else "ru",
                          subj=m.group(1).strip(), level=m.group(2).strip(), subset=r["Subset"]))
    return items


# ================================================================== inference
def infer(key, batch_size=16, token_budget=6000):
    """Batches are built by a token budget (batch size x longest prompt), so long questions go one by one."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    hf_id = MODELS[key][0]
    cap = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
    # bf16 is native only on Ampere+ (capability >= 8); on T4 use fp16, except Gemma 2, which overflows in fp16
    dtype = torch.bfloat16 if (cap >= 8 or "gemma" in hf_id.lower()) else torch.float16
    tok = AutoTokenizer.from_pretrained(hf_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    extra = {"attn_implementation": "eager"} if "gemma" in hf_id.lower() else {}
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:  # leave head-room on GPU 0, which holds the embeddings and the activations of the first layers
        extra["max_memory"] = {0: "9GiB", **{i: "13GiB" for i in range(1, n_gpu)}}
    model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=dtype, device_map="auto", **extra).eval()
    print(f"{key}: dtype={dtype}, devices={set(str(p.device) for p in model.parameters())}", flush=True)
    ids = [[i for i in {tok.encode(v, add_special_tokens=False)[-1] for v in (L, " " + L)}] for L in LETTERS]
    items = load_items()
    t0 = time.time()
    prompts = []
    for it in items:
        body = it["q"] + "\n" + "\n".join(f"{LETTERS[k]}. {o}" for k, o in enumerate(it["opts"]))
        msgs = [{"role": "user", "content": INSTR[it["lang"]] + "\n\n" + body}]
        try:
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            p = msgs[0]["content"] + "\nAnswer:"
        prompts.append(p)
    lens = np.array([len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts])
    print(f"{key}: {len(items)} questions, prompt length median {int(np.median(lens))}, max {lens.max()} tokens", flush=True)
    order = np.argsort(-lens)
    batches, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * lens[cur[0]] > token_budget or len(cur) >= batch_size:
            batches.append(cur); cur = []
        cur.append(i)
    batches.append(cur)
    logits = np.full((len(items), len(LETTERS)), -np.inf, dtype=np.float32)

    def run(idx):
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model(**enc, logits_to_keep=1)  # only the last position: avoids a batch x length x vocabulary tensor
        z = torch.log_softmax(out.logits[:, -1].float(), -1).cpu().numpy()
        for row, i in enumerate(idx):
            K = len(items[i]["opts"])
            logits[i, :K] = [np.logaddexp.reduce(z[row, ids[k]]) for k in range(K)]

    done, next_report = 0, 0
    for idx in batches:
        try:
            run(idx)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            for i in idx:  # fall back to one question at a time
                run([i])
                torch.cuda.empty_cache()
        done += len(idx)
        if done >= next_report:
            el = time.time() - t0
            print(f"{key}: {done}/{len(items)}  ({el/60:.0f} min elapsed, ~{el/done*(len(items)-done)/60:.0f} min left)", flush=True)
            next_report += 1000
    os.makedirs(OUT, exist_ok=True)
    bad = int((~np.isfinite(logits[:, 0])).sum())
    print(f"{key}: finished, rows with invalid logits: {bad}", flush=True)
    import transformers, platform
    with open(f"{OUT}/{key}_meta.json", "w") as f:
        json.dump({"gpu": (f"{n_gpu} x " if n_gpu > 1 else "") + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"),
                   "invalid_rows": bad, "hours": round((time.time() - t0) / 3600, 2), "python": platform.python_version(),
                   "torch": torch.__version__, "transformers": transformers.__version__, "dtype": str(dtype),
                   "prompt_len_median": int(np.median(lens)), "prompt_len_max": int(lens.max())}, f)
    np.savez_compressed(f"{OUT}/{key}.npz", logits=logits, y=np.array([it["y"] for it in items]),
                        lang=np.array([it["lang"] for it in items]), subj=np.array([it["subj"] for it in items]),
                        level=np.array([it["level"] for it in items]), nopt=np.array([len(it["opts"]) for it in items]))


# =================================================================== metrics
def softmax(z, T=1.0):
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(z, y):
    from scipy.optimize import minimize_scalar
    nll = lambda lt: -np.mean(np.log(softmax(z, np.exp(lt))[np.arange(len(y)), y] + 1e-12))
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def ece(p, y):
    conf, acc = p.max(1), (p.argmax(1) == y).astype(float)
    b = np.minimum((conf * N_BINS).astype(int), N_BINS - 1)
    return float(sum(abs(acc[b == k].mean() - conf[b == k].mean()) * (b == k).mean() for k in range(N_BINS) if (b == k).any()))


def brier(p, y):
    oh = np.zeros_like(p); oh[np.arange(len(y)), y] = 1
    return float(np.mean(((p - oh) ** 2).sum(1)))


def cq(s, alpha):
    n = len(s)
    return np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")


def lac_sets(pc, yc, pt):
    return ((1 - pt) <= cq(1 - pc[np.arange(len(yc)), yc], ALPHA)) & (pt > 0)


def aps_all(p):
    o = np.argsort(-p, 1)
    return np.take_along_axis(np.cumsum(np.take_along_axis(p, o, 1), 1), np.argsort(o, 1), 1)


def aps_sets(pc, yc, pt):
    return (aps_all(pt) <= cq(aps_all(pc)[np.arange(len(yc)), yc], ALPHA)) & (pt > 0)


def risk_coverage(p, y):
    o = np.argsort(-p.max(1))
    err = np.cumsum(p.argmax(1)[o] != y[o]) / np.arange(1, len(y) + 1)
    return np.arange(1, len(y) + 1) / len(y), err, float(err.mean())


def sgr_threshold(pc, yc, r_star):
    """Selection with guaranteed risk (Geifman & El-Yaniv, 2017), Clopper-Pearson bound."""
    from scipy.stats import beta
    conf, wrong = pc.max(1), pc.argmax(1) != yc
    o = np.argsort(-conf); conf, wrong = conf[o], wrong[o]
    m = len(conf); k = int(np.ceil(np.log2(m))); d = DELTA / k
    lo, hi, best = 0, m - 1, np.inf
    for _ in range(k):
        mid = (lo + hi) // 2; n = mid + 1; e = int(wrong[:n].sum())
        bound = 1.0 if e >= n else beta.ppf(1 - d, e + 1, n - e)
        if bound <= r_star:
            best, lo = conf[mid], mid + 1
        else:
            hi = mid - 1
        if lo > hi:
            break
    return best


def split(subj, seed):
    rng = np.random.default_rng(seed); cal = np.zeros(len(subj), bool)
    for s in np.unique(subj):
        i = np.where(subj == s)[0]; rng.shuffle(i); cal[i[: len(i) // 2]] = True
    return cal


# ================================================================== analysis
def run_scope(data, scope):
    rows, keep = [], {}
    for key, d in data.items():
        for seed in SEEDS:
            r = {"model": key, "seed": seed}; part = {}
            for lg in ["kk", "ru"]:
                m = (d["lang"] == lg) & d[scope]
                z, y, s = d["logits"][m], d["y"][m], d["subj"][m]
                cal = split(s, seed)
                T = fit_temperature(z[cal], y[cal])
                p0, p1, pc = softmax(z[~cal]), softmax(z[~cal], T), softmax(z[cal], T)
                yt, yc = y[~cal], y[cal]
                part[lg] = (pc, yc, p1, yt, z[~cal], T)
                acc = float((p1.argmax(1) == yt).mean())
                r.update({f"{lg}_n_cal": int(cal.sum()), f"{lg}_n_test": int((~cal).sum()), f"{lg}_acc": acc, f"{lg}_T": T,
                          f"{lg}_ece_raw": ece(p0, yt), f"{lg}_ece_ts": ece(p1, yt),
                          f"{lg}_brier_raw": brier(p0, yt), f"{lg}_brier_ts": brier(p1, yt),
                          f"{lg}_overconf": float(p0.max(1).mean() - acc)})
                for nm, fn in [("lac", lac_sets), ("aps", aps_sets)]:
                    C = fn(pc, yc, p1)
                    r[f"{lg}_{nm}_cov"] = float(C[np.arange(len(yt)), yt].mean())
                    r[f"{lg}_{nm}_size"] = float(C.sum(1).mean())
                    r[f"{lg}_{nm}_single"] = float((C.sum(1) == 1).mean())
                cov, err, r[f"{lg}_aurc"] = risk_coverage(p1, yt)
                for rs in TARGET_RISKS:
                    sel = p1.max(1) >= sgr_threshold(pc, yc, rs)
                    r[f"{lg}_cov@{rs}"] = float(sel.mean())
                    r[f"{lg}_risk@{rs}"] = float((p1[sel].argmax(1) != yt[sel]).mean()) if sel.any() else 0.0
                if seed == 0:
                    keep[(key, lg)] = dict(p0=p0, p1=p1, y=yt, s=s[~cal], cov=cov, err=err)
            # cross-lingual: temperature and threshold both estimated on Russian, applied to Kazakh
            pc_ru, yc_ru, _, _, _, T_ru = part["ru"]; _, _, _, y_kk, z_kk, _ = part["kk"]
            C = lac_sets(pc_ru, yc_ru, softmax(z_kk, T_ru))
            r["ru2kk_lac_cov"] = float(C[np.arange(len(y_kk)), y_kk].mean())
            rows.append(r)
    return rows, keep


def boot_diff(a, b, f, rng):
    """95% bootstrap CI of f(Kazakh) - f(Russian); the two samples are independent."""
    (pa, ya), (pb, yb) = a, b
    out = [f(pa[i], ya[i]) - f(pb[j], yb[j]) for i, j in
           ((rng.integers(0, len(ya), len(ya)), rng.integers(0, len(yb), len(yb))) for _ in range(N_BOOT))]
    return float(f(pa, ya) - f(pb, yb)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def analyze():
    import pandas as pd
    os.makedirs(f"{OUT}/tables", exist_ok=True)
    data = {}
    for key in MODELS:
        f = f"{OUT}/{key}.npz"
        if os.path.exists(f):
            d = dict(np.load(f, allow_pickle=True))
            both = set(d["subj"][d["lang"] == "kk"]) & set(d["subj"][d["lang"] == "ru"])
            hs = np.char.find(d["level"].astype(str), "High School") >= 0
            d["matched"] = np.isin(d["subj"], list(both)) & hs
            d["full"] = np.ones(len(d["y"]), bool)
            data[key] = d
    keys = list(data)
    assert keys, "no results/*.npz found - run the infer stage first"
    d0 = data[keys[0]]
    rows_m, keep_m = run_scope(data, "matched")
    rows_f, keep_f = run_scope(data, "full")
    dm, df_ = pd.DataFrame(rows_m), pd.DataFrame(rows_f)
    mean_m, sd_m = dm.groupby("model").mean(numeric_only=True), dm.groupby("model").std(numeric_only=True)
    mean_f = df_.groupby("model").mean(numeric_only=True)
    mean_m.to_csv(f"{OUT}/tables/matched_mean.csv"); sd_m.to_csv(f"{OUT}/tables/matched_sd.csv")
    mean_f.to_csv(f"{OUT}/tables/full_mean.csv")

    rng = np.random.default_rng(0)
    boot = {}
    for k in keys:
        a = (keep_m[(k, "kk")]["p0"], keep_m[(k, "kk")]["y"]); b = (keep_m[(k, "ru")]["p0"], keep_m[(k, "ru")]["y"])
        boot[k] = {"ece": boot_diff(a, b, ece, rng),
                   "acc": boot_diff(a, b, lambda p, y: float((p.argmax(1) == y).mean()), rng)}
    pd.DataFrame({k: {"d_ece": v["ece"][0], "d_ece_lo": v["ece"][1], "d_ece_hi": v["ece"][2],
                      "d_acc": v["acc"][0], "d_acc_lo": v["acc"][1], "d_acc_hi": v["acc"][2]} for k, v in boot.items()}).T \
        .to_csv(f"{OUT}/tables/bootstrap_kk_minus_ru.csv")

    # subject level (full scope, seed 0, all models pooled)
    sub = []
    for k in keys:
        for lg in ["kk", "ru"]:
            K = keep_f[(k, lg)]
            for s in np.unique(K["s"]):
                m = K["s"] == s
                if m.sum() >= 30:
                    sub.append(dict(model=k, lang=lg, subject=s, n=int(m.sum()),
                                    acc=float((K["p0"][m].argmax(1) == K["y"][m]).mean()), ece=ece(K["p0"][m], K["y"][m])))
    sub = pd.DataFrame(sub); sub.to_csv(f"{OUT}/tables/by_subject.csv", index=False)

    V = build_values(keys, d0, mean_m, sd_m, mean_f, dm, boot, sub)
    V = extra_values(V, keys)
    with open(f"{OUT}/results.json", "w", encoding="utf-8") as f:
        json.dump(V, f, ensure_ascii=False, indent=1)
    make_figures(keep_m, keys, V)
    print(json.dumps(V["flags"], indent=1))


def build_values(keys, d0, M, S, F, dm, boot, sub):
    pct = lambda x: f"{100 * x:.1f}"
    rng_ = lambda col, fmt, src=M: f"{fmt(src[col].min())}–{fmt(src[col].max())}"
    f3, f2 = (lambda x: f"{x:.3f}"), (lambda x: f"{x:.2f}")
    name = lambda k: MODELS[k][1]
    V = {"models": [name(k) for k in keys], "model_keys": keys, "n_models": len(keys)}
    # Table 1
    t1 = {}
    for lg in ["kk", "ru"]:
        m = d0["lang"] == lg
        t1[lg] = dict(total=f"{m.sum():,}", subsets=str(len(set(zip(d0["subj"][m], d0["level"][m])))),
                      hs=pct((np.char.find(d0["level"][m].astype(str), "High School") >= 0).mean()),
                      opts=f"{d0['nopt'][m].min()}–{d0['nopt'][m].max()}" if d0['nopt'][m].min() != d0['nopt'][m].max() else str(d0['nopt'][m].min()),
                      matched=f"{(m & d0['matched']).sum():,}",
                      cal=f"{int(M[f'{lg}_n_cal'].iloc[0]):,}", test=f"{int(M[f'{lg}_n_test'].iloc[0]):,}")
    V["t1"] = t1
    V["n_matched_subj"] = str(len(set(d0["subj"][d0["matched"]])))
    V["matched_subj_list"] = ", ".join(sorted(set(d0["subj"][d0["matched"]])))
    # Table 2-4 rows
    V["t2"] = [[name(k), lg, pct(M.loc[k, f"{lg}_acc"]), f3(M.loc[k, f"{lg}_ece_raw"]), f3(M.loc[k, f"{lg}_ece_ts"]),
                f3(M.loc[k, f"{lg}_brier_raw"]), f3(M.loc[k, f"{lg}_brier_ts"]), f2(M.loc[k, f"{lg}_T"])]
               for k in keys for lg in ["kk", "ru"]]
    for row in V["t2"]:
        row[1] = "Kazakh" if row[1] == "kk" else "Russian"
    V["t3"] = [[name(k), pct(M.loc[k, "kk_lac_cov"]), f2(M.loc[k, "kk_lac_size"]), pct(M.loc[k, "ru_lac_cov"]),
                f2(M.loc[k, "ru_lac_size"]), pct(M.loc[k, "ru2kk_lac_cov"])] for k in keys]
    V["t4"] = [[name(k), f3(M.loc[k, "kk_aurc"]), f3(M.loc[k, "ru_aurc"]), pct(M.loc[k, "kk_cov@0.1"]), pct(M.loc[k, "ru_cov@0.1"]),
                pct(M.loc[k, "kk_cov@0.2"]), pct(M.loc[k, "ru_cov@0.2"])] for k in keys]
    # text values
    best_kk, worst_kk = M["kk_acc"].idxmax(), M["kk_acc"].idxmin()
    best_ru = M["ru_acc"].idxmax()
    sel_kk = M["kk_cov@0.1"].idxmax(); sel_ru = M["ru_cov@0.1"].idxmax()
    oc = M["kk_overconf"]
    V.update({
        "acc_kk_min": pct(M.kk_acc.min()), "acc_kk_max": pct(M.kk_acc.max()), "acc_kk_min_model": name(worst_kk), "acc_kk_max_model": name(best_kk),
        "acc_ru_min": pct(M.ru_acc.min()), "acc_ru_max": pct(M.ru_acc.max()),
        "acc_kk_range": rng_("kk_acc", lambda x: f"{100*x:.0f}"), "acc_ru_range": rng_("ru_acc", lambda x: f"{100*x:.0f}"),
        "ece_kk_range": rng_("kk_ece_raw", f3), "ece_ru_range": rng_("ru_ece_raw", f3),
        "ece_kk_ts_range": rng_("kk_ece_ts", f3), "ece_ru_ts_range": rng_("ru_ece_ts", f3),
        "T_kk_range": rng_("kk_T", f2), "T_ru_range": rng_("ru_T", f2),
        "n_overconf_kk": str(int((oc > 0).sum())), "max_overconf_kk": f"{100*oc.max():.1f}", "max_overconf_model": name(oc.idxmax()),
        "acc_full_kk_range": rng_("kk_acc", pct, F), "acc_full_ru_range": rng_("ru_acc", pct, F),
        "ece_full_kk_range": rng_("kk_ece_raw", f3, F), "ece_full_ru_range": rng_("ru_ece_raw", f3, F),
        "cov_lac_range": f"{pct(min(M.kk_lac_cov.min(), M.ru_lac_cov.min()))}–{pct(max(M.kk_lac_cov.max(), M.ru_lac_cov.max()))}",
        "size_kk_range": rng_("kk_lac_size", f2), "size_ru_range": rng_("ru_lac_size", f2),
        "size_kk_mean": f"{M.kk_lac_size.mean():.1f}", "size_ru_mean": f"{M.ru_lac_size.mean():.1f}",
        "single_kk": f"{100*M.kk_lac_single.mean():.0f}", "single_ru": f"{100*M.ru_lac_single.mean():.0f}",
        "aps_extra": f2((M.kk_aps_size - M.kk_lac_size).mean() / 2 + (M.ru_aps_size - M.ru_lac_size).mean() / 2),
        "aps_cov_range": f"{pct(min(M.kk_aps_cov.min(), M.ru_aps_cov.min()))}–{pct(max(M.kk_aps_cov.max(), M.ru_aps_cov.max()))}",
        "xling_range": rng_("ru2kk_lac_cov", pct),
        "aurc_kk_range": rng_("kk_aurc", f3), "aurc_ru_range": rng_("ru_aurc", f3),
        "sel_kk_model": name(sel_kk), "sel_kk_cov": pct(M.loc[sel_kk, "kk_cov@0.1"]), "sel_kk_risk": pct(M.loc[sel_kk, "kk_risk@0.1"]),
        "sel_kk_abstain": f"{100*(1-M.loc[sel_kk, 'kk_cov@0.1']):.0f}",
        "sel_ru_cov": pct(M.loc[sel_ru, "ru_cov@0.1"]), "sel_kk_cov20": pct(M.loc[M['kk_cov@0.2'].idxmax(), "kk_cov@0.2"]),
        "risk_ok": str(int((dm[dm.model == sel_kk]["kk_risk@0.1"] <= 0.10).sum())), "n_seeds": str(len(SEEDS)),
    })
    # subject level
    skk = sub[sub.lang == "kk"].groupby("subject")[["acc", "ece"]].mean().sort_values("ece")
    V["corr_kk"] = f"{np.corrcoef(sub[sub.lang=='kk'].acc, sub[sub.lang=='kk'].ece)[0,1]:.2f}"
    V["corr_ru"] = f"{np.corrcoef(sub[sub.lang=='ru'].acc, sub[sub.lang=='ru'].ece)[0,1]:.2f}"
    V["subj_kk_worst"] = " and ".join(skk.index[-2:][::-1]); V["subj_kk_best"] = " and ".join(skk.index[:2])
    kzs = sub[sub.subject.isin(KZ_SPECIFIC)]; oth = sub[~sub.subject.isin(KZ_SPECIFIC)]
    V["ece_kzspec"], V["ece_other"] = f3(kzs.ece.mean()), f3(oth.ece.mean())
    # decisions (pre-registered rules, described in the Methods section)
    sig_worse = [k for k in keys if boot[k]["ece"][1] > 0]
    sig_better = [k for k in keys if boot[k]["ece"][2] < 0]
    V["n_sig_worse"], V["n_sig_better"] = str(len(sig_worse)), str(len(sig_better))
    sgn = lambda x: f"{x:.3f}".replace("-", "–")
    V["d_ece_min"] = sgn(min(b["ece"][0] for b in boot.values())); V["d_ece_max"] = sgn(max(b["ece"][0] for b in boot.values()))
    metas = [json.load(open(f"{OUT}/{k}_meta.json")) for k in keys if os.path.exists(f"{OUT}/{k}_meta.json")]
    if metas:
        V["gpu"] = ", ".join(sorted({m["gpu"] for m in metas}))
        V["gpu_hours"] = f"{np.mean([m['hours'] for m in metas]):.1f}"
        V["sw"] = f"Python {metas[0]['python']}, PyTorch {metas[0]['torch'].split('+')[0]}, Transformers {metas[0]['transformers']}"
    t_diff = [k for k in keys if abs(M.loc[k, "kk_T"] - M.loc[k, "ru_T"]) > 2 * max(S.loc[k, "kk_T"], S.loc[k, "ru_T"])]
    kaz = [k for k in keys if MODELS[k][2]]; base = "llama31" if "llama31" in keys else None
    V["flags"] = {
        "H_confirmed": len(sig_worse) > len(keys) / 2,
        "H_partial": 0 < len(sig_worse) <= len(keys) / 2,
        "overconf_kk_best": bool(M.loc[best_kk, "kk_overconf"] > 0),
        "kk_dev_larger": bool(abs(M.loc[best_kk, "kk_overconf"]) > abs(M.loc[best_kk, "ru_overconf"])),
        "T_differs": len(t_diff) > len(keys) / 2,
        "xling_under": bool(M.ru2kk_lac_cov.mean() < 1 - ALPHA - 0.01),
        "kaz_better_cal": bool(base and kaz and all(M.loc[k, "kk_ece_raw"] < M.loc[base, "kk_ece_raw"] for k in kaz)),
        "kaz_better_aurc": bool(base and kaz and all(M.loc[k, "kk_aurc"] < M.loc[base, "kk_aurc"] for k in kaz)),
        "size_larger_kk": bool(M.kk_lac_size.mean() > M.ru_lac_size.mean()),
        "sel_zero": bool(M["kk_cov@0.1"].max() == 0),
        "has_sig_better": len(sig_better) > 0,
    }
    V["fig2_model"] = name(best_kk)
    adapted = [k for k in keys if MODELS[k][2]]
    b = min(adapted, key=lambda k: M.loc[k, "kk_ece_raw"]) if adapted else best_kk
    V["fig2_models"] = [name("llama31" if "llama31" in keys else M["kk_ece_raw"].idxmax()), name(b)]
    return V


def extra_values(V, keys):
    """Per-model values quoted in the text of the article."""
    import pandas as pd
    M = pd.read_csv(f"{OUT}/tables/matched_mean.csv", index_col=0)
    B = pd.read_csv(f"{OUT}/tables/bootstrap_kk_minus_ru.csv", index_col=0)
    f3 = lambda x: f"{x:.3f}".replace("-", "–"); pct = lambda x: f"{100 * x:.1f}"
    E = {}
    if "llama31" in keys:
        d = np.load(f"{OUT}/llama31.npz", allow_pickle=True)
        for lg in ["kk", "ru"]:
            m = d["lang"] == lg
            E[f"llamaD_{lg}"] = f"{100 * (d['logits'][m].argmax(1) == 3).mean():.0f}"
        E["d_answerD"] = f"{100 * (d['y'] == 3).mean():.0f}"
        E["llama_err_kk"] = pct(1 - M.loc["llama31", "kk_acc"])
    for k in keys:
        for c in ["kk_acc", "ru_acc"]: E[f"{k}_{c}"] = pct(M.loc[k, c])
        for c in ["kk_ece_raw", "ru_ece_raw", "kk_ece_ts", "kk_aurc", "ru_aurc"]: E[f"{k}_{c}"] = f3(M.loc[k, c])
        for c in ["kk_T", "ru_T", "kk_lac_size", "ru_lac_size"]: E[f"{k}_{c}"] = f"{M.loc[k, c]:.2f}"
        for c in ["kk_cov@0.1", "ru_cov@0.1", "kk_cov@0.2", "ru2kk_lac_cov", "kk_lac_single", "ru_lac_single", "kk_risk@0.1"]:
            E[f"{k}_{c}"] = pct(M.loc[k, c])
        E[f"{k}_dece"] = f3(B.loc[k, "d_ece"]); E[f"{k}_dece_ci"] = f"{f3(B.loc[k, 'd_ece_lo'])} to {f3(B.loc[k, 'd_ece_hi'])}"
        E[f"{k}_dacc"] = f"{100 * B.loc[k, 'd_acc']:.1f}".replace("-", "–")
        E[f"{k}_dacc_ci"] = f"{100 * B.loc[k, 'd_acc_lo']:.1f} to {100 * B.loc[k, 'd_acc_hi']:.1f}".replace("-", "–")
    metas = {k: json.load(open(f"{OUT}/{k}_meta.json")) for k in keys if os.path.exists(f"{OUT}/{k}_meta.json")}
    if metas:
        E["prompt_median"] = {k: m.get("prompt_len_median") for k, m in metas.items()}
        h16 = [m["hours"] for m in metas.values() if "float16" in m["dtype"] and "bfloat16" not in m["dtype"]]
        E["hours16"] = f"{min(h16):.1f}–{max(h16):.1f}" if h16 else ""
        E["hours_gemma"] = f"{metas['gemma2']['hours']:.1f}" if "gemma2" in metas else ""
    V["E"] = E
    return V


def make_figures(keep, keys, V):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": ["Times New Roman", "Liberation Serif"], "font.size": 8,
                         "axes.titleweight": "normal", "axes.labelweight": "normal"})
    pair = [keys[V["models"].index(n)] for n in V["fig2_models"]]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9))
    for ax, mk, lab in zip(axes, pair, ["a", "b"]):
        K = keep[(mk, "kk")]
        for p, st, nm in [(K["p0"], "o-", "1"), (K["p1"], "s--", "2")]:
            conf, acc = p.max(1), p.argmax(1) == K["y"]
            b = np.minimum((conf * N_BINS).astype(int), N_BINS - 1)
            ok = [k for k in range(N_BINS) if (b == k).sum() >= 20]
            ax.plot([conf[b == k].mean() for k in ok], [acc[b == k].mean() for k in ok], st, color="black", ms=3, lw=0.8, label=nm)
        ax.plot([0, 1], [0, 1], ":", color="gray", lw=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence", labelpad=10); ax.set_ylabel("Accuracy", labelpad=10)
        ax.set_title(lab, fontsize=8); ax.legend(frameon=False)
    plt.tight_layout(); plt.savefig(f"{OUT}/Fig2_reliability.jpg", dpi=600, pil_kwargs={"quality": 95}); plt.close()
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), sharey=True)
    st = ["-", "--", ":", "-.", "-"]
    mk = ["o", "s", "^", "D", "v"]
    for ax, lg, lab in zip(axes, ["kk", "ru"], ["a", "b"]):
        for i, k in enumerate(keys):
            cov, err = keep[(k, lg)]["cov"], keep[(k, lg)]["err"]
            ax.plot(cov, err, linestyle=st[i % 5], marker=mk[i % 5], markevery=max(1, len(cov) // 8), ms=3.5,
                    mfc="white", color="black", lw=0.8, label=str(i + 1))
        ax.set_xlabel("Coverage", labelpad=10); ax.set_title(lab, fontsize=8); ax.legend(frameon=False)
    axes[0].set_ylabel("Selective error", labelpad=10)
    plt.tight_layout(); plt.savefig(f"{OUT}/Fig3_risk_coverage.jpg", dpi=600, pil_kwargs={"quality": 95}); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["infer", "analyze"])
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--batch_size", type=int, default=16)
    a = ap.parse_args()
    infer(a.model, a.batch_size) if a.stage == "infer" else analyze()
