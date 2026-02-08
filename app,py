import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.signal import find_peaks, savgol_filter
import re
import io
import math
import tempfile
import zipfile
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any, List

import plotly.graph_objects as go

# Optional baseline (sparse AsLS)
from scipy import sparse
from scipy.sparse.linalg import spsolve


# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="SERS Plotter v10", layout="wide")

st.title("Generátor SERS/EC-SERS spekter pro publikace 🧪")
st.caption("v10: univerzální workflow (s/bez napětí), tabulka pro labely/skupiny, batch export do ZIP, baseline/normalizace/resampling.")


# ----------------------------
# UTILITIES
# ----------------------------

@dataclass
class Meta:
    filename: str
    stem: str
    group_key: str
    potential_mV: Optional[float] = None
    step_mV: Optional[float] = None


def safe_stem(name: str) -> str:
    return re.sub(r"\.[^.]+$", "", name)


def parse_meta_from_filename(filename: str) -> Meta:
    """
    Robustně vytáhne potenciál/step z konce názvu, pokud existuje.
    group_key = název bez trailing *_<step>mV_<pot>mV nebo *_<pot>mV
    Funguje i pro ne-EC (mV neexistuje).
    """
    stem = safe_stem(filename)

    tokens = stem.split("_")
    potential = None
    step = None

    def is_mv(tok: str) -> bool:
        return re.fullmatch(r"-?\d+(?:\.\d+)?mV", tok) is not None

    # parse from end: ... _50mV_-100mV  OR ..._-100mV
    tmp = tokens[:]
    if tmp and is_mv(tmp[-1]):
        potential = float(tmp[-1].replace("mV", ""))
        tmp.pop()
        if tmp and is_mv(tmp[-1]):
            step = float(tmp[-1].replace("mV", ""))
            tmp.pop()

    group_key = "_".join(tmp) if tmp else stem

    return Meta(
        filename=filename,
        stem=stem,
        group_key=group_key,
        potential_mV=potential,
        step_mV=step
    )


@st.cache_data(show_spinner=False)
def load_txt_bytes(file_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load 2-column spectra from bytes (txt). Robust delimiters.
    """
    # Try whitespace/semicolon/comma
    df = pd.read_csv(io.BytesIO(file_bytes), sep=r"[,\t; ]+", header=None, engine="python")
    df = df.iloc[:, :2]
    df.columns = ["x", "y"]
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna().sort_values("x")
    return df["x"].values.astype(float), df["y"].values.astype(float)


def try_load_wdf(uploaded_file) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Attempts to read .wdf using one of supported libraries.
    Returns (x, y, msg) or None if not possible.
    """
    # Save to temp path because many readers want a file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    # Try renishawWiRE
    try:
        import renishawWiRE as rw
        w = rw.WDFReader(tmp_path)
        x = np.array(w.xdata, dtype=float)
        y = np.array(w.spectra[0], dtype=float)  # first spectrum by default
        idx = np.argsort(x)
        return x[idx], y[idx], "OK (.wdf přes renishawWiRE)"
    except Exception:
        pass

    # Try renishaw-wdf
    try:
        from renishaw_wdf import WDF
        w = WDF(tmp_path)
        x = np.array(w.x, dtype=float)
        y = np.array(w.spectra[0], dtype=float)
        idx = np.argsort(x)
        return x[idx], y[idx], "OK (.wdf přes renishaw-wdf)"
    except Exception:
        pass

    # Try RamanSPy
    try:
        import ramanspy as rp
        spec = rp.load.renishaw(tmp_path)
        x = np.array(spec.spectral_axis, dtype=float)
        y = np.array(spec.spectra[0], dtype=float)
        idx = np.argsort(x)
        return x[idx], y[idx], "OK (.wdf přes ramanspy)"
    except Exception:
        pass

    return None


def baseline_asls(y: np.ndarray, lam: float = 1e6, p: float = 1e-3, niter: int = 10) -> np.ndarray:
    """
    Asymmetric least squares baseline (sparse).
    """
    L = y.size
    if L < 3:
        return np.zeros_like(y)
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(L - 2, L))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * (D.T @ D)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return np.asarray(z)


def resample_to_grid(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    # simple linear interp is stable for spectra; avoids extra deps
    return np.interp(x_new, x, y)


def normalize_y(x: np.ndarray, y: np.ndarray, mode: str, peak_center: float = 1082, peak_window: float = 5) -> np.ndarray:
    mode = (mode or "none").lower()
    if mode == "none":
        return y
    if mode == "max":
        m = np.max(y)
        return y / (m if m != 0 else 1.0)
    if mode == "vector":
        n = np.linalg.norm(y)
        return y / (n if n != 0 else 1.0)
    if mode == "peak":
        m = (x >= peak_center - peak_window) & (x <= peak_center + peak_window)
        denom = np.max(y[m]) if np.any(m) else np.max(y)
        return y / (denom if denom != 0 else 1.0)
    return y


def find_nearest_idx(array: np.ndarray, value: float) -> int:
    array = np.asarray(array)
    return int((np.abs(array - value)).argmin())


def local_max_near(x: np.ndarray, y: np.ndarray, target: float, tol: float = 4.0) -> Optional[Tuple[float, float]]:
    m = (x >= target - tol) & (x <= target + tol)
    if not np.any(m):
        return None
    xi = x[m]; yi = y[m]
    j = int(np.argmax(yi))
    return float(xi[j]), float(yi[j])


def build_zip(files: Dict[str, bytes]) -> bytes:
    """
    files: { "name.ext": bytes }
    """
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, b in files.items():
            zf.writestr(name, b)
    bio.seek(0)
    return bio.getvalue()


# ----------------------------
# SIDEBAR: INPUTS
# ----------------------------
st.sidebar.header("0. Vstup")
uploaded_files = st.file_uploader(
    "Nahrajte spektra (.txt, případně .wdf)",
    type=["txt", "wdf"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Nahraj soubory a vpravo se objeví tabulka pro skupiny/labely + náhled a export.")
    st.stop()


# ----------------------------
# BUILD SPECTRA INDEX TABLE
# ----------------------------
rows = []
wdf_status_msgs = []
for f in uploaded_files:
    meta = parse_meta_from_filename(f.name)

    # default label:
    # - if potential exists -> "<pot> mV"
    # - else -> filename stem (or ???)
    default_label = f"{int(meta.potential_mV)} mV" if meta.potential_mV is not None else meta.stem

    rows.append({
        "include": True,
        "file": f.name,
        "group": meta.group_key,              # editable
        "label": default_label,               # editable
        "potential_mV": meta.potential_mV,    # read-only-ish
        "step_mV": meta.step_mV,
        "type": f.name.split(".")[-1].lower()
    })

df_index = pd.DataFrame(rows)

# Quick “EC mode?” detection: if at least 2 spectra have potential_mV
ec_detected = df_index["potential_mV"].notna().sum() >= 2


# ----------------------------
# SIDEBAR: WORKFLOW MODE + TABLE CONTROLS
# ----------------------------
st.sidebar.header("1. Workflow")

mode = st.sidebar.radio(
    "Režim exportu",
    ["Single figure (cokoliv vybereš)", "Batch po skupinách (ZIP)"],
    index=1 if ec_detected else 0
)

with st.sidebar.expander("Rychlé filtry", expanded=True):
    group_choices = sorted(df_index["group"].unique().tolist())
    selected_groups = st.multiselect("Skupiny (group)", options=group_choices, default=group_choices)

    label_contains = st.text_input("Label/file obsahuje (regex)", value="")
    filter_every_mV = st.number_input("EC filtr: kreslit jen každých N mV (0 = vypnout)", value=100 if ec_detected else 0, step=10)

    # Apply filters to defaults (we don't destroy the table; just help user)
    df_filtered = df_index.copy()
    df_filtered = df_filtered[df_filtered["group"].isin(selected_groups)]

    if label_contains.strip():
        try:
            rx = re.compile(label_contains.strip())
            df_filtered = df_filtered[
                df_filtered["label"].astype(str).str.contains(rx) |
                df_filtered["file"].astype(str).str.contains(rx)
            ]
        except re.error:
            st.warning("Neplatný regex, filtr ignoruju.")

    # EC downsample suggestion
    if filter_every_mV and filter_every_mV > 0 and df_filtered["potential_mV"].notna().any():
        pot = df_filtered["potential_mV"].astype(float)
        keep = np.isclose(np.mod(np.abs(pot), float(filter_every_mV)), 0.0, atol=1e-6)
        # if keep would drop everything, keep all
        if keep.any():
            df_filtered.loc[~keep, "include"] = False


st.sidebar.header("2. Úpravy dat")
with st.sidebar.expander("Předzpracování", expanded=True):
    # Cropping + x handling
    x_min = st.number_input("X min (cm⁻¹)", value=300, step=10)
    x_max = st.number_input("X max (cm⁻¹)", value=1800, step=10)
    invert_x = st.checkbox("Invertovat osu X", value=False)

    # Smoothing
    do_smooth = st.checkbox("Savitzky–Golay smoothing", value=True)
    sg_window = st.slider("SG okno (liché)", 5, 51, 11, step=2)
    sg_poly = st.slider("SG polynom", 2, 5, 3)

    # Baseline
    do_baseline = st.checkbox("Baseline korekce (AsLS)", value=False)
    asls_lam = st.number_input("AsLS λ", value=1.0e6, format="%.2e")
    asls_p = st.number_input("AsLS p", value=1.0e-3, format="%.2e")
    asls_niter = st.slider("AsLS iterace", 5, 30, 10)

    # Resampling
    do_resample = st.checkbox("Resampling na společnou osu (doporučeno pro stabilní píky)", value=True)
    x_step = st.number_input("Krok osy (cm⁻¹)", value=1.0, step=0.5)

    # Normalize
    norm_mode = st.selectbox("Normalizace", ["none", "max", "vector", "peak"], index=3)
    peak_center = st.number_input("Peak center (cm⁻¹) pro 'peak' normalizaci", value=1082.0, step=1.0)
    peak_window = st.number_input("Peak window (± cm⁻¹)", value=5.0, step=1.0)


st.sidebar.header("3. Vzhled a export")
with st.sidebar.expander("Rozměry a kvalita", expanded=True):
    col_w, col_h = st.columns(2)
    with col_w:
        img_width_px = st.number_input("Šířka (px)", value=1200, step=100)
    with col_h:
        img_height_px = st.number_input("Výška (px)", value=1000, step=100)
    img_dpi = st.number_input("DPI", value=150, step=50)

    figsize_w = img_width_px / img_dpi
    figsize_h = img_height_px / img_dpi
    st.caption(f"Matplotlib figsize ≈ {figsize_w:.2f} × {figsize_h:.2f} in")

with st.sidebar.expander("Grafika", expanded=False):
    palette_name = st.selectbox("Paleta", ["jet", "viridis", "plasma", "inferno", "coolwarm", "bwr", "rainbow"], index=0)
    offset_val = st.number_input("Offset (posun Y)", value=2000, step=100)

    xlabel_text = st.text_input("Popis osy X", "Ramanův posun (cm⁻¹)")
    ylabel_text = st.text_input("Popis osy Y", "Intenzita (a.u.)")

    line_width = st.slider("Tloušťka čáry", 0.5, 3.0, 1.5)
    font_size = st.slider("Velikost písma os", 8, 30, 14)

    # Optional “right axis label + arrow” for EC style
    show_right_label = st.checkbox("Zobrazit pravý popis osy (např. Potenciál)", value=ec_detected)
    right_label_text = st.text_input("Text vpravo", "Potenciál")
    show_arrow = st.checkbox("Šipka (směr série)", value=ec_detected)

    export_svg = st.checkbox("Export SVG", value=True)
    export_png = st.checkbox("Export PNG", value=True)
    export_pdf = st.checkbox("Export PDF", value=False)

with st.sidebar.expander("Fonty pro Illustrator", expanded=False):
    font_family = st.text_input("Font family", value="Arial")
    svg_fonttype = st.selectbox("SVG: text jako text", ["none", "path"], index=0)
    pdf_fonttype = st.selectbox("PDF fonttype", ["42 (TrueType)", "3 (Type3)"], index=0)

st.sidebar.header("4. Píky")
with st.sidebar.expander("Detekce a popisky", expanded=False):
    peak_label_size = st.slider("Velikost písma popisků", 8, 30, 14)
    label_height_offset = st.slider("Výška popisků nad píkem", 50, 5000, 500, step=50)
    show_peak_lines = st.checkbox("Vodící čáry", value=True)

    peak_mode = st.selectbox("Režim píků", ["žádné", "auto", "fixní seznam + doladění"], index=2)
    prominence = st.slider("Auto: prominence", 10, 5000, 200)

    fixed_peak_list_str = st.text_area(
        "Fixní seznam píků (cm⁻¹), oddělené čárkou",
        value="1613,1598,1515,1497,1398,1365,1291,1230,1211,1182,1076,839,787,747,721,651,639,588,521,469,409,348,239"
    )
    fixed_tol = st.slider("Tolerance (± cm⁻¹) pro fixní píky", 1, 20, 4)

    manual_add_str = st.text_input("➕ Přidat píky (např. 1001, 1580)", value="")
    manual_remove_str = st.text_input("➖ Smazat píky (např. 220)", value="")

def parse_peak_list(s: str) -> List[float]:
    out = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            pass
    return out

manual_adds = parse_peak_list(manual_add_str)
manual_removes = parse_peak_list(manual_remove_str)
fixed_peaks = parse_peak_list(fixed_peak_list_str) if peak_mode == "fixní seznam + doladění" else []


# ----------------------------
# MAIN: EDIT TABLE (most practical)
# ----------------------------
st.subheader("1) Tabulka spekter (edituj group/label/include)")
st.write("Tip: pro univerzálnost je nejrychlejší upravit labely/skupiny přímo tady. Funguje i pro spektra bez napětí.")

edited = st.data_editor(
    df_filtered,
    use_container_width=True,
    num_rows="fixed",
    column_config={
        "include": st.column_config.CheckboxColumn("include"),
        "group": st.column_config.TextColumn("group"),
        "label": st.column_config.TextColumn("label"),
        "potential_mV": st.column_config.NumberColumn("potential_mV", format="%.0f"),
        "step_mV": st.column_config.NumberColumn("step_mV", format="%.0f"),
        "file": st.column_config.TextColumn("file"),
        "type": st.column_config.TextColumn("type"),
    },
    disabled=["file", "type", "potential_mV", "step_mV"],
    key="spectra_table"
)

selected_df = edited[edited["include"] == True].copy()
if selected_df.empty:
    st.warning("Nemáš vybrané žádné spektrum (include = False).")
    st.stop()


# ----------------------------
# LOAD + PROCESS SELECTED
# ----------------------------
def load_spectrum(uploaded_file) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Returns x, y, status
    """
    suffix = uploaded_file.name.split(".")[-1].lower()
    if suffix == "txt":
        x, y = load_txt_bytes(uploaded_file.getvalue())
        return x, y, "OK (.txt)"
    if suffix == "wdf":
        got = try_load_wdf(uploaded_file)
        if got is None:
            return np.array([]), np.array([]), "NELZE načíst .wdf (nainstaluj renishawWiRE / renishaw-wdf / ramanspy)"
        x, y, msg = got
        return x, y, msg
    return np.array([]), np.array([]), "Nepodporovaný formát"


def preprocess(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # crop
    m = (x >= min(x_min, x_max)) & (x <= max(x_min, x_max))
    if np.any(m):
        x = x[m]
        y = y[m]

    # sort (just in case)
    idx = np.argsort(x)
    x = x[idx]; y = y[idx]

    # baseline
    if do_baseline and len(y) > 10:
        bl = baseline_asls(y, lam=float(asls_lam), p=float(asls_p), niter=int(asls_niter))
        y = y - bl

    # smoothing
    if do_smooth and len(y) > sg_window:
        y = savgol_filter(y, int(sg_window), int(sg_poly))

    return x, y


def build_common_grid(all_x: List[np.ndarray]) -> Optional[np.ndarray]:
    if not do_resample:
        return None
    xmin = max(np.min(x) for x in all_x)
    xmax = min(np.max(x) for x in all_x)
    if xmin >= xmax:
        return None
    step = float(x_step) if float(x_step) > 0 else 1.0
    return np.arange(math.floor(xmin), math.ceil(xmax) + step, step)


# Map filename -> UploadedFile object
upload_map = {f.name: f for f in uploaded_files}

# Load all selected spectra
loaded = []
status_lines = []
for row in selected_df.itertuples(index=False):
    uf = upload_map.get(row.file)
    if uf is None:
        continue
    x, y, status = load_spectrum(uf)
    if x.size == 0 or y.size == 0:
        status_lines.append(f"❌ {row.file}: {status}")
        continue
    x, y = preprocess(x, y)
    loaded.append({
        "file": row.file,
        "group": row.group,
        "label": row.label,
        "potential_mV": row.potential_mV,
        "x": x,
        "y": y
    })
    status_lines.append(f"✅ {row.file}: {status}")

with st.expander("Stav načtení souborů", expanded=False):
    st.write("\n".join(status_lines))

if not loaded:
    st.error("Nepovedlo se načíst žádná data.")
    st.stop()

# Build common x grid (optional)
common_x = build_common_grid([d["x"] for d in loaded])

# Normalize + resample
for d in loaded:
    x, y = d["x"], d["y"]
    if common_x is not None and len(common_x) > 10:
        y = resample_to_grid(x, y, common_x)
        x = common_x
    y = normalize_y(x, y, mode=norm_mode, peak_center=float(peak_center), peak_window=float(peak_window))
    d["x"], d["y"] = x, y


# ----------------------------
# PREVIEW: PLOTLY (interactive)
# ----------------------------
with st.expander("🔍 Interaktivní náhled (Plotly)", expanded=False):
    fig_int = go.Figure()
    cmap = plt.get_cmap(palette_name)
    colors = cmap(np.linspace(0, 1, len(loaded)))
    plotly_colors = [mcolors.to_hex(c) for c in colors]

    # Order: if potentials exist -> sort by potential desc, else by label
    has_pot = any(d["potential_mV"] is not None and not (pd.isna(d["potential_mV"])) for d in loaded)
    if has_pot:
        loaded_sorted = sorted(loaded, key=lambda z: float(z["potential_mV"]) if z["potential_mV"] is not None else -1e9, reverse=True)
    else:
        loaded_sorted = sorted(loaded, key=lambda z: str(z["label"]))
    for i, d in enumerate(loaded_sorted):
        x = d["x"]; y = d["y"] + i * float(offset_val)
        fig_int.add_trace(go.Scatter(x=x, y=y, mode="lines", name=str(d["label"]), line=dict(color=plotly_colors[i])))

    fig_int.update_layout(
        height=500,
        xaxis_title=xlabel_text,
        yaxis_title=ylabel_text,
        hovermode="x unified",
        template="plotly_dark",
        xaxis=dict(autorange="reversed" if invert_x else True)
    )
    st.plotly_chart(fig_int, use_container_width=True)


# ----------------------------
# PEAKS: compute once for top trace (or per-figure later)
# ----------------------------
def compute_peak_positions(x: np.ndarray, y: np.ndarray) -> List[float]:
    """
    Returns list of x positions to label.
    """
    positions = []

    if peak_mode == "žádné":
        return positions

    if peak_mode == "auto":
        idxs, _ = find_peaks(y, prominence=float(prominence), distance=30)
        positions.extend([float(x[i]) for i in idxs])

    if peak_mode == "fixní seznam + doladění":
        # Place each fixed peak at local maximum near target
        for pk in fixed_peaks:
            hit = local_max_near(x, y, pk, tol=float(fixed_tol))
            if hit:
                positions.append(hit[0])

    # Manual adds: snap to local maximum near target
    for pk in manual_adds:
        idx = find_nearest_idx(x, pk)
        w = 10
        s = max(0, idx - w); e = min(len(x), idx + w + 1)
        best = s + int(np.argmax(y[s:e]))
        positions.append(float(x[best]))

    # Manual removes: remove labels close to any remove target
    cleaned = []
    for pos in positions:
        if any(abs(pos - r) < 15 for r in manual_removes):
            continue
        cleaned.append(pos)

    # Unique + sort (descending x looks nicer when invert_x, but we keep ascending and let axis invert)
    cleaned = sorted(set([round(p, 3) for p in cleaned]))
    return cleaned


# ----------------------------
# FINAL PLOT (Matplotlib) helper
# ----------------------------
def make_matplotlib_figure(curves: List[Dict[str, Any]], title: str = "") -> plt.Figure:
    # Illustrator-friendly settings
    plt.rcParams["font.family"] = font_family
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["svg.fonttype"] = svg_fonttype
    plt.rcParams["pdf.fonttype"] = 42 if "42" in pdf_fonttype else 3

    fig, ax = plt.subplots(figsize=(figsize_w, figsize_h), dpi=img_dpi)
    ax.set_xlabel(xlabel_text)
    ax.set_ylabel(ylabel_text)
    ax.set_yticks([])

    if title:
        ax.set_title(title)

    cmap = plt.get_cmap(palette_name)
    mpl_colors = cmap(np.linspace(0, 1, len(curves)))

    # ordering: potential desc if available else keep given order
    has_pot = any(c.get("potential_mV") is not None and not pd.isna(c.get("potential_mV")) for c in curves)
    if has_pot:
        curves = sorted(curves, key=lambda z: float(z["potential_mV"]) if z["potential_mV"] is not None else -1e9, reverse=True)

    # compute peak labels from top curve (after sorting)
    top = curves[0]
    top_x = top["x"]
    top_y = top["y"] + (0 * 0)  # no offset yet
    peak_positions = compute_peak_positions(top_x, top["y"])

    for i, c in enumerate(curves):
        x = c["x"]
        y = c["y"] + i * float(offset_val)
        ax.plot(x, y, color=mpl_colors[i], lw=float(line_width))

        # Right-side label (always useful; for EC it's like your templates)
        trans = ax.get_yaxis_transform()
        y_lbl = y[0] if invert_x else y[-1]
        ax.text(
            1.01, y_lbl, str(c["label"]),
            color=mpl_colors[i],
            va="center", ha="left",
            fontsize=font_size, fontweight="bold",
            transform=trans, clip_on=False
        )

        # Peak labels only on top trace (i == 0)
        if i == 0 and peak_positions and peak_mode != "žádné":
            for px in peak_positions:
                idx = find_nearest_idx(x, px)
                py = y[idx]
                if show_peak_lines:
                    ax.plot([px, px], [py + 50, py + float(label_height_offset) - 50],
                            color="black", lw=0.5, alpha=0.8)
                ax.text(px, py + float(label_height_offset), f"{int(round(px))}",
                        rotation=90, ha="center", va="bottom",
                        fontsize=int(peak_label_size))

    # Axis range + invert
    ax.set_xlim(x_max, x_min) if invert_x else ax.set_xlim(x_min, x_max)

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Optional right label + arrow (EC-style)
    if show_right_label:
        axr = ax.twinx()
        axr.set_ylabel(right_label_text)
        axr.set_yticks([])
        axr.spines["top"].set_visible(False)
        axr.spines["left"].set_visible(False)
        axr.spines["right"].set_visible(False)

    if show_arrow:
        ax.annotate(
            "", xy=(1.02, 0.05), xytext=(1.02, 0.95),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", lw=3),
        )

    fig.tight_layout()
    return fig


def fig_to_bytes(fig: plt.Figure, fmt: str) -> bytes:
    bio = io.BytesIO()
    fig.savefig(bio, format=fmt, bbox_inches="tight", dpi=img_dpi)
    bio.seek(0)
    return bio.getvalue()


# ----------------------------
# EXPORT / DISPLAY
# ----------------------------
st.subheader("2) Finální výstup")

# Create curve list (single selection)
curves_all = loaded

if mode.startswith("Single"):
    fig = make_matplotlib_figure(curves_all)
    st.pyplot(fig)

    col1, col2, col3 = st.columns(3)

    if export_svg:
        svg_bytes = fig_to_bytes(fig, "svg")
        col1.download_button("📥 SVG (vektor)", svg_bytes, file_name="SERS_output.svg", mime="image/svg+xml")

    if export_png:
        png_bytes = fig_to_bytes(fig, "png")
        col2.download_button(f"📥 PNG ({img_width_px}×{img_height_px})", png_bytes, file_name="SERS_output.png", mime="image/png")

    if export_pdf:
        pdf_bytes = fig_to_bytes(fig, "pdf")
        col3.download_button("📥 PDF (vektor)", pdf_bytes, file_name="SERS_output.pdf", mime="application/pdf")

    plt.close(fig)

else:
    # Batch by group
    st.write("Batch režim: pro každou skupinu (group) se vyrobí samostatná figura a stáhneš ZIP.")

    grouped = {}
    for d in curves_all:
        grouped.setdefault(d["group"], []).append(d)

    # Generate figures and zip
    out_files: Dict[str, bytes] = {}
    preview_groups = list(grouped.keys())[:3]  # show first few previews
    for gname, curves in grouped.items():
        fig = make_matplotlib_figure(curves, title=str(gname))
        if export_svg:
            out_files[f"{gname}.svg"] = fig_to_bytes(fig, "svg")
        if export_png:
            out_files[f"{gname}.png"] = fig_to_bytes(fig, "png")
        if export_pdf:
            out_files[f"{gname}.pdf"] = fig_to_bytes(fig, "pdf")
        plt.close(fig)

    zip_bytes = build_zip(out_files)

    st.download_button(
        "📦 Stáhnout ZIP se všemi figurami",
        data=zip_bytes,
        file_name="SERS_figures.zip",
        mime="application/zip"
    )

    with st.expander("Rychlý preview (první 3 skupiny)", expanded=False):
        for gname in preview_groups:
            st.markdown(f"**{gname}**")
            fig = make_matplotlib_figure(grouped[gname], title=str(gname))
            st.pyplot(fig)
            plt.close(fig)


# ----------------------------
# FOOTER: .wdf help
# ----------------------------
with st.expander("ℹ️ .wdf podpora (pokud chceš)", expanded=False):
    st.markdown(
        """
Pokud chceš načítat **.wdf** přímo z WiRE, doinstaluj jednu z knihoven:

- `pip install renishawWiRE`
- nebo `pip install renishaw-wdf`
- nebo `pip install ramanspy`

Appka poběží i bez toho – jen pro `.txt`.
"""
    )
