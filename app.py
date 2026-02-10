import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.ndimage import minimum_filter
import re
import io
import json
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# --- KONFIGURACE STRÁNKY ---
st.set_page_config(page_title="SERS Plotter v12 - Universal", layout="wide")

st.title("🧪 Univerzální Generátor SERS Spekter pro Publikace")
st.markdown("""
**v13.0**: Zjednodušený program - pouze 2 režimy (Napětí a Obecná spektra)
""")

# --- FUNKCE PRO BASELINE KOREKCI ---

def baseline_als(y, lam=1e6, p=0.01, niter=10):
    """
    Asymmetric Least Squares (ALS) baseline korekce
    Parametry:
        y: spektrum
        lam: smoothness (větší = hladší baseline)
        p: asymmetry (menší = více se přizpůsobí minimům)
        niter: počet iterací
    """
    L = len(y)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    W = diags(w, 0, shape=(L, L))
    
    for i in range(niter):
        W.setdiag(w)
        Z = W + D
        z = spsolve(csc_matrix(Z), w*y)
        w = p * (y > z) + (1-p) * (y < z)
    
    return z

def baseline_polynomial(y, degree=3):
    """
    Polynomiální baseline korekce
    """
    x = np.arange(len(y))
    coeffs = np.polyfit(x, y, degree)
    baseline = np.polyval(coeffs, x)
    return baseline

def baseline_rolling_ball(y, window_size=50):
    """
    Rolling ball baseline korekce
    """
    baseline = minimum_filter(y, size=window_size, mode='constant')
    return baseline

def normalize_spectrum(y, x=None, method='max'):
    """
    Normalizace spektra
    method:
      - 'max'    : maximum = 1
      - 'area'   : plocha = 1 (integrál |y| dx)
      - 'minmax' : 0-1
    """
    y = np.asarray(y)

    if method == 'max':
        max_val = np.max(y)
        return y / max_val if max_val != 0 else y

    elif method == 'area':
        ay = np.abs(y)
        if x is None:
            area = float(np.sum((ay[:-1] + ay[1:]) * 0.5))
        else:
            x = np.asarray(x)
            dx = np.diff(x)
            if len(dx) != len(ay) - 1:
                area = float(np.sum((ay[:-1] + ay[1:]) * 0.5))
            else:
                area = float(np.sum((ay[:-1] + ay[1:]) * 0.5 * dx))
        return y / area if area != 0 else y

    elif method == 'minmax':
        min_val, max_val = np.min(y), np.max(y)
        return (y - min_val) / (max_val - min_val) if (max_val - min_val) != 0 else y

    return y
def get_voltage_from_filename(filename):
    """Vytáhne poslední číslo před 'mV'."""
    matches = re.findall(r'([-\d]+)mV', filename)
    if matches:
        return int(matches[-1])
    return None

def detect_scan_direction(filename):
    """Rozpozná směr skenu podle klíčového slova v názvu."""
    filename_lower = filename.lower()
    if "reverse" in filename_lower or "zp" in filename_lower or "back" in filename_lower:
        return "reverse"
    return "forward"

def load_data(uploaded_file):
    """Načte data z txt souboru."""
    try:
        uploaded_file.seek(0)
        
        # Pokus o načtení s různými separátory
        try:
            # Pokus 1: mezera nebo tabulátor
            df = pd.read_csv(uploaded_file, sep=r'\s+', header=None, engine='python')
        except:
            # Pokus 2: čárka
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=',', header=None, engine='python')
        
        # Validace - musí mít alespoň 2 sloupce
        if df.shape[1] < 2:
            st.error(f"Soubor {uploaded_file.name} nemá dostatečný počet sloupců (potřeba min. 2)")
            return None, None
        
        # Vezmeme první dva sloupce
        df = df.iloc[:, :2]
        df.columns = ['x', 'y']
        
        # Konverze na numerické hodnoty
        df['x'] = pd.to_numeric(df['x'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        
        # Odstranění NaN
        df = df.dropna()
        
        # Kontrola jestli máme data
        if len(df) == 0:
            st.error(f"Soubor {uploaded_file.name} neobsahuje platná numerická data")
            return None, None
        
        # Seřazení podle X
        df = df.sort_values(by='x')
        
        return df['x'].values, df['y'].values
        
    except Exception as e:
        st.error(f"Chyba při načítání souboru {uploaded_file.name}: {str(e)}")
        return None, None

def find_nearest_idx(array, value):
    """Najde nejbližší index k dané hodnotě."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def generate_label(item, mode, custom_label=None):
    """Generuje popisek podle zvoleného režimu."""
    if custom_label:
        return custom_label
    
    if mode == "voltage":
        return item.get('label', item['filename'])
    elif mode == "filename":
        return Path(item['filename']).stem
    elif mode == "number":
        return f"Spektrum {item.get('index', 1)}"
    else:
        return item['filename']

# --- FUNKCE PRO ŠABLONY ---

def save_template(settings, name):
    """Uloží nastavení jako šablonu."""
    if 'templates' not in st.session_state:
        st.session_state.templates = {}
    
    template = {
        'name': name,
        'created': datetime.now().isoformat(),
        'settings': settings
    }
    st.session_state.templates[name] = template
    return True

def load_template(name):
    """Načte šablonu."""
    if 'templates' not in st.session_state or name not in st.session_state.templates:
        return None
    return st.session_state.templates[name]['settings']

def get_current_settings():
    """Získá aktuální nastavení z session_state."""
    settings = {}
    
    # Projde všechny klíče v session_state a uloží relevantní
    for key in st.session_state:
        if not key.startswith('_') and key not in ['templates', 'custom_labels', 'custom_order']:
            settings[key] = st.session_state[key]
    
    return settings

def export_templates_json():
    """Exportuje všechny šablony do JSON."""
    if 'templates' not in st.session_state:
        return None
    
    return json.dumps(st.session_state.templates, indent=2)

def import_templates_json(json_str):
    """Importuje šablony z JSON."""
    try:
        templates = json.loads(json_str)
        if 'templates' not in st.session_state:
            st.session_state.templates = {}
        st.session_state.templates.update(templates)
        return True
    except Exception as e:
        st.error(f"Chyba při importu: {e}")
        return False

# --- SESSION STATE PRO PERZISTENCI ---
if 'custom_labels' not in st.session_state:
    st.session_state.custom_labels = {}
if 'custom_order' not in st.session_state:
    st.session_state.custom_order = []
if 'templates' not in st.session_state:
    st.session_state.templates = {}

# --- SPRÁVA ŠABLON (HORNÍ LIŠTA) ---
col_title, col_template = st.columns([3, 1])

with col_template:
    with st.popover("💾 Šablony", use_container_width=True):
        st.subheader("Správa šablon")
        
        # Uložit novou šablonu
        with st.expander("➕ Uložit aktuální nastavení", expanded=False):
            template_name = st.text_input("Název šablony:", "")
            if st.button("💾 Uložit šablonu"):
                if template_name:
                    settings = get_current_settings()
                    save_template(settings, template_name)
                    st.success(f"✅ Šablona '{template_name}' uložena!")
                else:
                    st.warning("⚠️ Zadejte název šablony")
        
        # Načíst existující šablonu
        if st.session_state.templates:
            st.divider()
            st.write("**📂 Uložené šablony:**")
            
            for template_name in st.session_state.templates:
                col1, col2 = st.columns([3, 1])
                with col1:
                    template_info = st.session_state.templates[template_name]
                    created = template_info.get('created', 'N/A')[:10]
                    st.text(f"📄 {template_name} ({created})")
                with col2:
                    if st.button("📥", key=f"load_{template_name}", help="Načíst"):
                        settings = load_template(template_name)
                        if settings:
                            for key, value in settings.items():
                                st.session_state[key] = value
                            st.rerun()
        
        # Export/Import
        st.divider()
        with st.expander("📤 Export/Import šablon", expanded=False):
            if st.session_state.templates:
                json_export = export_templates_json()
                st.download_button(
                    "📥 Stáhnout všechny šablony (JSON)",
                    json_export,
                    "sers_templates.json",
                    "application/json"
                )
            
            uploaded_templates = st.file_uploader("📤 Načíst šablony ze souboru", type=['json'])
            if uploaded_templates:
                json_str = uploaded_templates.read().decode('utf-8')
                if import_templates_json(json_str):
                    st.success("✅ Šablony importovány!")
                    st.rerun()

# --- HLAVNÍ LOGIKA ---

# Režim aplikace
st.sidebar.header("🎯 Režim Práce")
work_mode = st.sidebar.radio(
    "Vyberte typ spekter:",
    ["📊 Spektra s napětím (série)", "📈 Obecná spektra (libovolná)"],
    help="Série = spektra měřená pod napětím, Obecná = jakákoliv spektra"
)

uploaded_files = st.file_uploader(
    "📁 Nahrajte .txt soubory spekter", 
    type=['txt'], 
    accept_multiple_files=True,
    help="Můžete nahrát libovolný počet souborů. Podporovány jsou .txt soubory s dvěma sloupci (x, y)."
)

if uploaded_files:
    
    # --- ZPRACOVÁNÍ NAHRANÝCH SOUBORŮ ---
    all_files_meta = []
    for idx, f in enumerate(uploaded_files):
        raw_volts = get_voltage_from_filename(f.name)
        direction = detect_scan_direction(f.name)
        
        all_files_meta.append({
            'file': f,
            'filename': f.name,
            'index': idx + 1,
            'raw_volts': raw_volts if raw_volts is not None else 0,
            'direction': direction,
            'has_voltage': raw_volts is not None
        })
    
    st.success(f"✅ Načteno {len(all_files_meta)} souborů")
    
    # --- BOČNÍ PANEL: REŽIM-SPECIFICKÁ NASTAVENÍ ---
    st.sidebar.header("1️⃣ Výběr a Úprava Spekter")
    
    # ======================
    # REŽIM 1: SPEKTRA S NAPĚTÍM
    # ======================
    if "napětím" in work_mode:
        with st.sidebar.expander("⚡ Nastavení napěťové série", expanded=True):
            # Filtr směru skenu
            scan_filter = st.radio(
                "Směr skenu:",
                ["Dopředný (Forward)", "Zpětný (Reverse)", "Všechny"],
                index=0
            )
            
            # Filtrace podle směru
            if scan_filter == "Dopředný (Forward)":
                current_batch = [x for x in all_files_meta if x['direction'] == 'forward']
            elif scan_filter == "Zpětný (Reverse)":
                current_batch = [x for x in all_files_meta if x['direction'] == 'reverse']
            else:
                current_batch = all_files_meta
            
            st.caption(f"📊 {len(current_batch)} souborů po filtraci")
            st.divider()
            
            # Záporné znamínko
            force_minus = st.checkbox(
                "Záporné hodnoty napětí (-)",
                value=True,
                help="Přidá mínus před všechny nenulové hodnoty (50 → -50)"
            )
            
            st.divider()
            
            # NOVÉ: Vlastní formát popisků
            st.write("**📝 Formát popisků:**")
            label_format_mode = st.radio(
                "Typ:",
                ["Jen hodnota (např. '-100 mV')", "Vlastní šablona", "Název souboru"],
                index=0,
                help="Jak se budou zobrazovat popisky spekter"
            )
            
            if label_format_mode == "Vlastní šablona":
                label_template = st.text_input(
                    "Šablona popisku:",
                    "{voltage} mV",
                    help="Použijte {voltage} pro hodnotu napětí, {filename} pro název souboru. Např: 'Vzorek {voltage}mV' nebo '{voltage}mV dopředný sken'"
                )
                
                # Ukázka
                sample_voltage = -100 if force_minus else 100
                sample_preview = label_template.replace("{voltage}", str(sample_voltage)).replace("{filename}", "sample.txt")
                st.caption(f"📋 Ukázka: {sample_preview}")
            else:
                label_template = None
            
            st.divider()
            
            # Řazení (stacking)
            stack_order = st.radio(
                "Pořadí spekter (shora dolů):",
                ["Od 0 do Max", "Od Max do 0"],
                help="Určuje, které spektrum bude nahoře a které dole"
            )
            
            # Rychlý výběr podle kroku
            auto_step = st.number_input(
                "Krok pro automatický výběr (mV)",
                value=100,
                step=10,
                help="Vybere pouze spektra s násobky této hodnoty"
            )
        
        # Zpracování napěťových dat
        processed_batch = []
        for item in current_batch:
            final_volts = item['raw_volts']
            if force_minus and final_volts > 0:
                final_volts = -final_volts
            
            new_item = item.copy()
            new_item['volts'] = final_volts
            
            # Generování popisku podle zvoleného formátu
            if "Název souboru" in label_format_mode:
                new_item['display_label'] = Path(item['filename']).stem
                new_item['label'] = new_item['display_label']
            elif "Vlastní šablona" in label_format_mode and label_template:
                label = label_template.replace("{voltage}", str(final_volts))
                label = label.replace("{filename}", Path(item['filename']).stem)
                new_item['display_label'] = label
                new_item['label'] = new_item['display_label']
            else:  # "Jen hodnota"
                new_item['display_label'] = f"{final_volts} mV"
                new_item['label'] = new_item['display_label']
            
            processed_batch.append(new_item)
        
        # Seřazení
        processed_batch.sort(key=lambda x: x['volts'])
        if stack_order == "Od Max do 0":
            processed_batch.reverse()
        
        # Reset selection pokud se změnil formát popisků
        current_label_key = f"{label_format_mode}_{force_minus}"
        if 'prev_label_format' not in st.session_state:
            st.session_state.prev_label_format = current_label_key
        
        if st.session_state.prev_label_format != current_label_key:
            st.session_state.prev_label_format = current_label_key
            if 'voltage_selection' in st.session_state:
                # Aktualizovat výběr s novými popisky
                st.session_state.voltage_selection = [s['label'] for s in processed_batch if abs(s['raw_volts']) % auto_step == 0]
        
        # Výběr spekter
        options = [s['label'] for s in processed_batch]
        default_selection = [s['label'] for s in processed_batch if abs(s['raw_volts']) % auto_step == 0]
        
        # Rychlé akce
        st.write("### 🎯 Výběr Spekter")
        col1, col2, col3, col4 = st.columns(4)
        
        select_all = col1.button("✅ Vybrat vše", use_container_width=True)
        select_none = col2.button("❌ Zrušit vše", use_container_width=True)
        select_step = col3.button(f"🎚️ Krok {auto_step}mV", use_container_width=True)
        invert_selection = col4.button("🔄 Invertovat", use_container_width=True)
        
        # Inicializace session state pro výběr
        if 'voltage_selection' not in st.session_state or select_all or select_none or select_step or invert_selection:
            if select_all:
                st.session_state.voltage_selection = options
            elif select_none:
                st.session_state.voltage_selection = []
            elif select_step:
                st.session_state.voltage_selection = default_selection
            elif invert_selection:
                current = st.session_state.get('voltage_selection', default_selection)
                st.session_state.voltage_selection = [opt for opt in options if opt not in current]
            else:
                st.session_state.voltage_selection = default_selection
        
        selected_labels = st.multiselect(
            "Zahrnout do grafu:",
            options=options,
            default=st.session_state.voltage_selection,
            help="Pořadí zde určuje pořadí v grafu (odspodu nahoru)",
            key=f"voltage_multiselect_{label_format_mode}_{force_minus}"
        )
        
        # Aktualizace session state
        st.session_state.voltage_selection = selected_labels
        
        final_data_list = [s for s in processed_batch if s['label'] in selected_labels]
    
    # ======================
    # REŽIM 2: OBECNÁ SPEKTRA
    # ======================
    elif "Obecná" in work_mode:
        with st.sidebar.expander("📋 Nastavení popisků", expanded=True):
            label_mode = st.radio(
                "Typ popisků:",
                ["Název souboru", "Číslo (Spektrum 1, 2, ...)", "Vlastní text"],
                help="Jak se budou označovat jednotlivá spektra"
            )
            
            # Řazení
            sort_mode = st.radio(
                "Řazení spekter:",
                ["Podle názvu souboru (A-Z)", "Podle pořadí nahrání", "Vlastní"],
                help="Jak budou spektra seřazena v grafu (odspodu nahoru)"
            )
        
        # Seřazení podle zvoleného módu
        if sort_mode == "Podle názvu souboru (A-Z)":
            all_files_meta.sort(key=lambda x: x['filename'])
        elif sort_mode == "Vlastní":
            st.sidebar.info("💡 Vlastní řazení: Přeuspořádejte pořadí v multiselect níže")
        
        # Generování popisků
        for idx, item in enumerate(all_files_meta):
            if label_mode == "Název souboru":
                item['display_label'] = Path(item['filename']).stem
            elif label_mode == "Číslo (Spektrum 1, 2, ...)":
                item['display_label'] = f"Spektrum {idx + 1}"
            else:  # Vlastní text
                if item['filename'] not in st.session_state.custom_labels:
                    st.session_state.custom_labels[item['filename']] = f"Spektrum {idx + 1}"
                item['display_label'] = st.session_state.custom_labels[item['filename']]
        
        # Vlastní popisky - editace
        if label_mode == "Vlastní text":
            with st.sidebar.expander("✏️ Editace vlastních popisků", expanded=False):
                for item in all_files_meta:
                    new_label = st.text_input(
                        f"📄 {item['filename'][:30]}...",
                        value=st.session_state.custom_labels[item['filename']],
                        key=f"label_{item['filename']}"
                    )
                    st.session_state.custom_labels[item['filename']] = new_label
                    item['display_label'] = new_label
        
        # Reset selection pokud se změnil typ popisků
        if 'prev_label_mode_general' not in st.session_state:
            st.session_state.prev_label_mode_general = label_mode
        
        if st.session_state.prev_label_mode_general != label_mode:
            st.session_state.general_selection = [item['display_label'] for item in all_files_meta]
            st.session_state.prev_label_mode_general = label_mode
        
        # Výběr spekter
        options = [item['display_label'] for item in all_files_meta]
        
        # Rychlé akce
        st.write("### 🎯 Výběr Spekter")
        col1, col2, col3 = st.columns(3)
        
        select_all_general = col1.button("✅ Vybrat vše", use_container_width=True, key="select_all_general")
        select_none_general = col2.button("❌ Zrušit vše", use_container_width=True, key="select_none_general")
        invert_general = col3.button("🔄 Invertovat", use_container_width=True, key="invert_general")
        
        # Inicializace session state
        if 'general_selection' not in st.session_state or select_all_general or select_none_general or invert_general:
            if select_all_general:
                st.session_state.general_selection = options
            elif select_none_general:
                st.session_state.general_selection = []
            elif invert_general:
                current = st.session_state.get('general_selection', options)
                st.session_state.general_selection = [opt for opt in options if opt not in current]
            else:
                st.session_state.general_selection = options
        
        if sort_mode == "Vlastní":
            st.info("💡 Vlastní řazení: Pořadí v seznamu níže určuje pořadí v grafu (odspodu nahoru). Přesuňte položky myší.")
        
        selected_labels = st.multiselect(
            "Zahrnout do grafu:",
            options=options,
            default=st.session_state.general_selection,
            help="Vyberte spektra a přeuspořádejte je tažením. Pořadí zde = pořadí v grafu odspodu nahoru.",
            key=f"general_multiselect_{label_mode}_{sort_mode}"
        )
        
        # Aktualizace session state
        st.session_state.general_selection = selected_labels
        
        # Zachování pořadí z multiselect (multiselect v Streamlit zachovává pořadí jak uživatel vybírá)
        label_to_item = {item['display_label']: item for item in all_files_meta}
        final_data_list = [label_to_item[label] for label in selected_labels if label in label_to_item]
    # --- NASTAVENÍ VZHLEDU ---
    st.sidebar.header("2️⃣ Vzhled a Export")
    
    with st.sidebar.expander("📏 Rozměry obrázku", expanded=False):
        preset = st.selectbox(
            "Předvolby:",
            ["Vlastní", "Publikace (1200×1000)", "Prezentace (1920×1080)", "Poster (2400×1800)"]
        )
        
        if preset == "Publikace (1200×1000)":
            img_width_px, img_height_px, img_dpi = 1200, 1000, 300
        elif preset == "Prezentace (1920×1080)":
            img_width_px, img_height_px, img_dpi = 1920, 1080, 150
        elif preset == "Poster (2400×1800)":
            img_width_px, img_height_px, img_dpi = 2400, 1800, 300
        else:
            col_w, col_h = st.columns(2)
            with col_w:
                img_width_px = st.number_input("Šířka (px)", value=1200, step=100)
            with col_h:
                img_height_px = st.number_input("Výška (px)", value=1000, step=100)
            img_dpi = st.number_input("DPI", value=300, step=50, help="Pro publikace doporučeno 300")
        
        figsize_w = img_width_px / img_dpi
        figsize_h = img_height_px / img_dpi
        
        st.caption(f"📐 Výsledná velikost: {figsize_w:.2f}\" × {figsize_h:.2f}\" @ {img_dpi} DPI")

    with st.sidebar.expander("🎨 Grafika a Osy", expanded=True):
        # Paleta barev
        col1, col2 = st.columns(2)
        with col1:
            palette_name = st.selectbox(
                "Paleta barev:",
                ["jet", "viridis", "plasma", "inferno", "magma", "coolwarm", "bwr", "rainbow", "turbo"],
                index=0
            )
        with col2:
            reverse_colors = st.checkbox("Obrátit paletu", value=False)
        
        st.divider()
        
        # Offset mezi spektry
        offset_val = st.number_input(
            "Offset mezi spektry (Y)",
            value=2000,
            step=100,
            help="Vertikální rozestup mezi spektry"
        )
        
        st.divider()
        
        # Popisky os
        col1, col2 = st.columns(2)
        with col1:
            xlabel_text = st.text_input("Osa X:", "Ramanův posun (cm⁻¹)")
        with col2:
            ylabel_text = st.text_input("Osa Y:", "Intenzita (a.u.)")
        
        # Rozsah X
        x_min_default, x_max_default = 300, 1800
        x_range = st.slider(
            "Rozsah osy X:",
            0, 4000,
            (x_min_default, x_max_default),
            help="Zobrazená část spektra"
        )
        
        # Další nastavení
        col1, col2 = st.columns(2)
        with col1:
            invert_x = st.checkbox("Invertovat X", value=False)
        with col2:
            show_grid = st.checkbox("Zobrazit mřížku", value=False)
        
        st.divider()
        
        # Styly čar
        col1, col2 = st.columns(2)
        with col1:
            line_width = st.slider("Tloušťka spekter:", 0.5, 5.0, 1.5, 0.1)
        with col2:
            font_size = st.slider("Velikost písma:", 8, 30, 14, 1)
        
        # Pokročilé
        with st.expander("⚙️ Pokročilé styly", expanded=False):
            axis_line_width = st.slider("Tloušťka os:", 0.5, 3.0, 1.5, 0.1)
            label_position = st.radio("Pozice popisků spekter:", ["Vpravo", "Uvnitř grafu"], index=0)
            smooth_spectra = st.checkbox("Vyhlazení spekter (Savitzky-Golay)", value=True)
            if smooth_spectra:
                smooth_window = st.slider("Okno vyhlazení:", 5, 21, 11, 2)
                smooth_poly = st.slider("Polynom řádu:", 1, 5, 3, 1)
    
    # --- BASELINE KOREKCE ---
    with st.sidebar.expander("🔬 Baseline Korekce", expanded=False):
        apply_baseline = st.checkbox(
            "Aplikovat baseline korekci",
            value=False,
            help="Odstraní fluorescenční pozadí ze spekter"
        )
        
        if apply_baseline:
            baseline_method = st.selectbox(
                "Metoda:",
                ["ALS (Asymmetric Least Squares)", "Polynom", "Rolling Ball"],
                help="ALS je nejvšestrannější, Polynom je rychlý, Rolling Ball pro jednoduché pozadí"
            )
            
            if "ALS" in baseline_method:
                st.info("💡 ALS je nejlepší pro fluorescenční pozadí")
                baseline_lam = st.slider(
                    "Vyhlazení (λ):",
                    100000, 10000000, 1000000, 100000,
                    help="Větší hodnota = hladší baseline",
                    format="%d"
                )
                baseline_p = st.slider(
                    "Asymetrie (p):",
                    0.001, 0.1, 0.01, 0.001,
                    help="Menší hodnota = více se přizpůsobí minimům",
                    format="%.3f"
                )
                baseline_niter = st.slider(
                    "Iterace:",
                    5, 20, 10, 1,
                    help="Více iterací = přesnější, ale pomalejší"
                )
            
            elif "Polynom" in baseline_method:
                baseline_degree = st.slider(
                    "Stupeň polynomu:",
                    1, 6, 3, 1,
                    help="Vyšší stupeň = složitější křivka baseline"
                )
            
            else:  # Rolling Ball
                baseline_window = st.slider(
                    "Velikost okna:",
                    10, 200, 50, 5,
                    help="Větší okno = hladší baseline"
                )
            
            # Náhled baseline
            show_baseline_preview = st.checkbox(
                "Zobrazit náhled baseline",
                value=False,
                help="Přidá do grafu samotnou baseline pro kontrolu"
            )
    
    # --- NORMALIZACE ---
    with st.sidebar.expander("📊 Normalizace Spekter", expanded=False):
        apply_normalization = st.checkbox(
            "Normalizovat spektra",
            value=False,
            help="Přizpůsobí všechna spektra na stejnou velikost"
        )
        
        if apply_normalization:
            norm_method = st.selectbox(
                "Metoda normalizace:",
                ["Maximum = 1", "Plocha = 1", "Min-Max (0-1)"],
                help="Maximum: nejjednodušší, Plocha: pro kvantitativní porovnání, Min-Max: celý rozsah 0-1"
            )
            
            norm_scale = st.slider(
                "Škálování po normalizaci:",
                100, 10000, 1000, 100,
                help="Násobitel pro lepší vizualizaci"
            )
            
            st.info("💡 Normalizace se aplikuje před offsetem mezi spektry")

    # --- SPRÁVA PÍKŮ ---
    st.sidebar.header("3️⃣ Správa Píků")
    
    with st.sidebar.expander("📍 Nastavení píků", expanded=True):
        peak_target = st.radio(
            "Zobrazit píky u:",
            ["Nejvyšší spektrum", "Nejnižší spektrum", "Všechna spektra", "Konkrétní spektrum", "Vypnuto"],
            index=0
        )
        
        if peak_target == "Konkrétní spektrum" and final_data_list:
            peak_spectrum_idx = st.selectbox(
                "Vyberte spektrum:",
                range(len(final_data_list)),
                format_func=lambda x: final_data_list[x].get('display_label', final_data_list[x]['filename'])
            )
        
        st.divider()
        
        # Detekce píků
        col1, col2 = st.columns(2)
        with col1:
            use_auto_peaks = st.checkbox("Auto-detekce", value=True)
        with col2:
            show_peak_lines = st.checkbox("Vodící čáry", value=True)
        
        if use_auto_peaks:
            prominence = st.slider(
                "Citlivost detekce:",
                10, 2000, 100, 10,
                help="Vyšší hodnota = méně píků"
            )
            min_distance = st.slider(
                "Min. vzdálenost píků:",
                5, 100, 30, 5,
                help="Minimální vzdálenost mezi dvěma píky"
            )
        
        st.divider()
        
        # Manuální úpravy
        col1, col2 = st.columns(2)
        with col1:
            manual_add_str = st.text_input(
                "➕ Přidat píky:",
                "",
                help="Oddělte čárkou, např: 1001, 1320, 1580"
            )
        with col2:
            manual_remove_str = st.text_input(
                "➖ Odstranit píky:",
                "",
                help="Oddělte čárkou, např: 220, 450"
            )
        
        # Styl píků
        with st.expander("🎨 Styl popisků píků", expanded=False):
            peak_label_size = st.slider("Velikost textu:", 8, 24, 12, 1)
            label_height_offset = st.slider("Výška nad píkem:", 50, 5000, 500, 50)
            peak_label_rotation = st.slider("Rotace textu:", 0, 90, 90, 15)
            peak_line_color = st.color_picker("Barva čar:", "#000000")
            peak_line_alpha = st.slider("Průhlednost čar:", 0.0, 1.0, 0.8, 0.1)

    # Zpracování manuálních úprav píků
    manual_adds = []
    manual_removes = []
    
    if manual_add_str:
        try:
            manual_adds = [int(float(x.strip())) for x in manual_add_str.split(',') if x.strip()]
        except ValueError:
            st.sidebar.error("❌ Neplatný formát pro přidání píků")
    
    if manual_remove_str:
        try:
            manual_removes = [int(float(x.strip())) for x in manual_remove_str.split(',') if x.strip()]
        except ValueError:
            st.sidebar.error("❌ Neplatný formát pro odstranění píků")

    # --- EXPORT A DÁVKOVÉ ZPRACOVÁNÍ ---
    st.sidebar.header("4️⃣ Export")
    
    with st.sidebar.expander("💾 Nastavení exportu", expanded=False):
        export_formats = st.multiselect(
            "Formáty k exportu:",
            ["SVG (vektorový)", "PNG (rastrový)", "PDF (tisk)"],
            default=["SVG (vektorový)", "PNG (rastrový)"]
        )
        
        auto_filename = st.checkbox("Automatický název souboru", value=True)
        if not auto_filename:
            custom_filename = st.text_input("Název souboru:", "SERS_output")
        else:
            custom_filename = f"SERS_{len(final_data_list)}spectra"

    # --- VYKRESLOVÁNÍ ---
    if final_data_list and len(final_data_list) > 0:
        
        # Příprava barev
        cmap = plt.get_cmap(palette_name)
        if reverse_colors:
            cmap = cmap.reversed()
        mpl_colors = cmap(np.linspace(0, 1, len(final_data_list)))
        plotly_colors = [mcolors.to_hex(c) for c in mpl_colors]
        
        # --- INTERAKTIVNÍ NÁHLED ---
        with st.expander("🔍 Interaktivní náhled (Plotly)", expanded=False):
            fig_int = go.Figure()
            
            for i, item in enumerate(final_data_list):
                x, y = load_data(item['file'])
                if x is None:
                    continue
                
                # Filtrace rozsahu
                mask = (x >= x_range[0]) & (x <= x_range[1])
                x_c, y_c = x[mask], y[mask]
                
                # Baseline korekce
                if apply_baseline:
                    if "ALS" in baseline_method:
                        baseline = baseline_als(y_c, lam=baseline_lam, p=baseline_p, niter=baseline_niter)
                    elif "Polynom" in baseline_method:
                        baseline = baseline_polynomial(y_c, degree=baseline_degree)
                    else:  # Rolling Ball
                        baseline = baseline_rolling_ball(y_c, window_size=baseline_window)
                    
                    y_c = y_c - baseline
                
                # Vyhlazení
                if smooth_spectra and len(y_c) > smooth_window:
                    y_c = savgol_filter(y_c, smooth_window, smooth_poly)
                
                # Normalizace
                if apply_normalization:
                    if "Maximum" in norm_method:
                        y_c = normalize_spectrum(y_c, method='max') * norm_scale
                    elif "Plocha" in norm_method:
                        y_c = normalize_spectrum(y_c, x=x_c, method='area') * norm_scale
                    else:  # Min-Max
                        y_c = normalize_spectrum(y_c, method='minmax') * norm_scale
                
                # Offset
                y_s = y_c + (i * offset_val)
                
                # Popisek
                label = item.get('display_label', item['filename'])
                
                fig_int.add_trace(go.Scatter(
                    x=x_c,
                    y=y_s,
                    mode='lines',
                    name=label,
                    line=dict(color=plotly_colors[i], width=line_width),
                    hovertemplate=f'<b>{label}</b><br>x=%{{x:.1f}}<br>y=%{{y:.1f}}<extra></extra>'
                ))
            
            fig_int.update_layout(
                height=600,
                xaxis_title=xlabel_text,
                yaxis_title=ylabel_text,
                hovermode="x unified",
                template="plotly_white",
                xaxis=dict(autorange="reversed" if invert_x else True),
                font=dict(size=font_size),
                showlegend=True,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            st.plotly_chart(fig_int, use_container_width=True)
        
        # --- FINÁLNÍ STATICKÝ GRAF ---
        st.subheader("📊 Finální Graf pro Export")
        
        # Nastavení matplotlib
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['svg.fonttype'] = 'none'
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.size'] = font_size
        plt.rcParams['axes.linewidth'] = axis_line_width
        
        # Vytvoření figury
        fig, ax = plt.subplots(figsize=(figsize_w, figsize_h), dpi=img_dpi)
        
        # Určení spektra pro píky
        peak_indices = []
        if peak_target == "Nejvyšší spektrum":
            peak_indices = [len(final_data_list) - 1]
        elif peak_target == "Nejnižší spektrum":
            peak_indices = [0]
        elif peak_target == "Všechna spektra":
            peak_indices = list(range(len(final_data_list)))
        elif peak_target == "Konkrétní spektrum" and 'peak_spectrum_idx' in locals():
            peak_indices = [peak_spectrum_idx]
        
        # Vykreslení spekter
        for i, item in enumerate(final_data_list):
            x, y = load_data(item['file'])
            if x is None:
                st.warning(f"⚠️ Nepodařilo se načíst soubor: {item['filename']}")
                continue
            
            # Filtrace rozsahu
            mask = (x >= x_range[0]) & (x <= x_range[1])
            x_c, y_c = x[mask], y[mask]
            
            # Baseline korekce
            baseline = None
            if apply_baseline:
                if "ALS" in baseline_method:
                    baseline = baseline_als(y_c, lam=baseline_lam, p=baseline_p, niter=baseline_niter)
                elif "Polynom" in baseline_method:
                    baseline = baseline_polynomial(y_c, degree=baseline_degree)
                else:  # Rolling Ball
                    baseline = baseline_rolling_ball(y_c, window_size=baseline_window)
                
                y_c = y_c - baseline
            
            # Vyhlazení (po baseline korekci)
            if smooth_spectra and len(y_c) > smooth_window:
                y_c = savgol_filter(y_c, smooth_window, smooth_poly)
            
            # Normalizace
            if apply_normalization:
                if "Maximum" in norm_method:
                    y_c = normalize_spectrum(y_c, method='max') * norm_scale
                elif "Plocha" in norm_method:
                    y_c = normalize_spectrum(y_c, x=x_c, method='area') * norm_scale
                else:  # Min-Max
                    y_c = normalize_spectrum(y_c, method='minmax') * norm_scale
            
            # Offset
            y_s = y_c + (i * offset_val)
            
            # Vykreslení spektra
            ax.plot(x_c, y_s, color=mpl_colors[i], lw=line_width)
            
            # Náhled baseline (pokud je zapnut)
            if apply_baseline and show_baseline_preview and baseline is not None:
                baseline_shifted = baseline + (i * offset_val)
                ax.plot(x_c, baseline_shifted, color=mpl_colors[i], lw=0.5, linestyle='--', alpha=0.5)
            
            # Popisek spektra
            label = item.get('display_label', item['filename'])
            
            if label_position == "Vpravo":
                trans = ax.get_yaxis_transform()
                y_lbl = y_s[0] if invert_x else y_s[-1]
                ax.text(
                    1.02, y_lbl, label,
                    color=mpl_colors[i],
                    va='center',
                    ha='left',
                    fontsize=font_size,
                    fontweight='bold',
                    transform=trans,
                    clip_on=False
                )
            else:  # Uvnitř grafu
                x_pos = x_c[-1] if not invert_x else x_c[0]
                ax.text(
                    x_pos, y_s[-1 if not invert_x else 0],
                    label,
                    color=mpl_colors[i],
                    va='center',
                    ha='right' if not invert_x else 'left',
                    fontsize=font_size,
                    fontweight='bold'
                )
            
            # Vykreslení píků
            if i in peak_indices:
                final_peaks = []
                
                # Automatická detekce
                if use_auto_peaks:
                    peaks, _ = find_peaks(y_s, prominence=prominence, distance=min_distance)
                    final_peaks.extend(peaks)
                
                # Manuální přidání
                for user_x in manual_adds:
                    idx = find_nearest_idx(x_c, user_x)
                    search_window = 10
                    start = max(0, idx - search_window)
                    end = min(len(x_c), idx + search_window)
                    
                    if start < end:
                        local_max_idx = start + np.argmax(y_s[start:end])
                        # Zamezení duplikátů
                        if not any(abs(existing - local_max_idx) < 5 for existing in final_peaks):
                            final_peaks.append(local_max_idx)
                
                # Odstranění manuálně vyřazených píků
                valid_peaks = [
                    p for p in final_peaks
                    if not any(abs(x_c[p] - remove_x) < 15 for remove_x in manual_removes)
                ]
                
                # Vykreslení označení píků
                for peak_idx in valid_peaks:
                    px, py = x_c[peak_idx], y_s[peak_idx]
                    
                    # Vodící čára
                    if show_peak_lines:
                        ax.plot(
                            [px, px],
                            [py + 50, py + label_height_offset - 50],
                            color=peak_line_color,
                            lw=0.5,
                            alpha=peak_line_alpha
                        )
                    
                    # Popisek píku
                    ax.text(
                        px,
                        py + label_height_offset,
                        f"{int(px)}",
                        rotation=peak_label_rotation,
                        ha='center',
                        va='bottom',
                        fontsize=peak_label_size,
                        color=peak_line_color
                    )
        
        # Nastavení os
        ax.set_xlabel(xlabel_text, fontweight='bold')
        ax.set_ylabel(ylabel_text, fontweight='bold')
        
        if invert_x:
            ax.set_xlim(x_range[1], x_range[0])
        else:
            ax.set_xlim(x_range[0], x_range[1])
        
        # Skrytí některých prvků
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yticks([])
        
        # Mřížka
        if show_grid:
            ax.grid(True, alpha=0.3, linestyle='--')
        
        # Zobrazení grafu
        st.pyplot(fig)
        
        # --- EXPORT ---
        st.subheader("💾 Stažení Výstupů")
        
        export_cols = st.columns(len(export_formats))
        
        for idx, fmt in enumerate(export_formats):
            with export_cols[idx]:
                if "SVG" in fmt:
                    svg_io = io.BytesIO()
                    plt.savefig(svg_io, format='svg', bbox_inches='tight', dpi=img_dpi)
                    svg_io.seek(0)
                    st.download_button(
                        "📥 Stáhnout SVG",
                        svg_io,
                        f"{custom_filename}.svg",
                        "image/svg+xml",
                        use_container_width=True
                    )
                
                elif "PNG" in fmt:
                    png_io = io.BytesIO()
                    plt.savefig(png_io, format='png', bbox_inches='tight', dpi=img_dpi)
                    png_io.seek(0)
                    st.download_button(
                        "📥 Stáhnout PNG",
                        png_io,
                        f"{custom_filename}.png",
                        "image/png",
                        use_container_width=True
                    )
                
                elif "PDF" in fmt:
                    pdf_io = io.BytesIO()
                    plt.savefig(pdf_io, format='pdf', bbox_inches='tight', dpi=img_dpi)
                    pdf_io.seek(0)
                    st.download_button(
                        "📥 Stáhnout PDF",
                        pdf_io,
                        f"{custom_filename}.pdf",
                        "application/pdf",
                        use_container_width=True
                    )
        
        plt.close(fig)
        
        # Statistiky
        with st.expander("📊 Statistiky a Metadata", expanded=False):
            st.write("### Informace o grafu")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Počet spekter", len(final_data_list))
            with col2:
                st.metric("Rozsah X", f"{x_range[0]}-{x_range[1]} cm⁻¹")
            with col3:
                st.metric("Offset", f"{offset_val} a.u.")
            
            st.write("### Seznam zpracovaných souborů")
            for i, item in enumerate(final_data_list):
                st.text(f"{i+1}. {item['filename']} → {item.get('display_label', 'N/A')}")
    
    elif uploaded_files and len(final_data_list) == 0:
        st.warning("⚠️ Nebyla vybrána žádná spektra k zobrazení. Upravte filtry nebo výběr v postranním panelu.")

else:
    # Uvítací obrazovka
    st.info("""
    ### 👋 Vítejte v SERS Plotter v13.0!
    
    **Jak začít:**
    1. Nahrajte .txt soubory s vašimi Ramanovými spektry
    2. Vyberte režim práce (napěťové série nebo obecná spektra)
    3. Použijte rychlá tlačítka (✅ Vybrat vše, ❌ Zrušit vše, 🔄 Invertovat)
    4. Nastavte vlastní formát popisků (volitelné)
    5. Aplikujte baseline korekci a normalizaci (podle potřeby)
    6. Upravte vzhled a označte píky
    7. Uložte nastavení jako šablonu
    8. Exportujte finální graf
    
    **Nové ve v13.0:**
    - 🎯 **Zjednodušení** - pouze 2 jasné režimy (Napětí a Obecná)
    - ⚡ **Rychlejší** - odstraněn složitý pokročilý režim
    - 🧹 **Přehlednější kód** - snazší údržba a ladění
    
    **Klíčové funkce:**
    - 💾 Šablony nastavení - ukládání a sdílení
    - 🔬 Baseline korekce (ALS, Polynom, Rolling Ball)
    - 📊 Normalizace spekter
    - 📝 Vlastní šablony popisků
    - 📍 Pokročilá správa píků
    - ⚡ Rychlé akce pro výběr spekter
    
    **Podporované formáty:**
    - `.txt` soubory se dvěma sloupci (x, y) oddělenými mezerou nebo tabulátorem
    
    **Tip:** Pro automatické popisky použijte šablonu "{voltage} mV" nebo přidejte vlastní text
    """)
    
    # Rychlá nápověda
    with st.expander("📚 Podrobná dokumentace", expanded=False):
        st.markdown("""
        ### Režimy práce
        
        **📊 Spektra s napětím:**
        - Optimalizováno pro napěťové série (0 mV až -1000 mV atd.)
        - Automatická detekce napětí z názvu souboru
        - Možnost filtrování dopředného/zpětného skenu
        - Rychlý výběr podle kroku (např. každých 100 mV)
        
        **📈 Obecná spektra:**
        - Pro jakákoliv spektra bez napětí
        - Flexibilní systém pojmenování
        - Vlastní nebo automatické popisky
        - Různé možnosti řazení
        
        ### Baseline korekce
        
        **ALS (Asymmetric Least Squares):**
        - Nejlepší pro fluorescenční pozadí
        - λ (lambda): Vyhlazení baseline - vyšší hodnota = hladší křivka
        - p (asymetrie): Jak moc se baseline přizpůsobí minimům vs. maximům
        - Doporučené hodnoty: λ=1,000,000, p=0.01
        
        **Polynomiální:**
        - Rychlá metoda pro jednoduché pozadí
        - Stupeň 2-3 pro mírně zakřivené pozadí, 4-6 pro složitější
        
        **Rolling Ball:**
        - Simuluje kulující se kouli pod spektrem
        - Vhodné pro jednoduché, plynulé pozadí
        
        ### Normalizace
        
        **Maximum = 1:**
        - Nejjednodušší metoda
        - Všechna spektra mají stejnou maximální intenzitu
        
        **Plocha = 1:**
        - Pro kvantitativní porovnání
        - Zachovává relativní intenzity píků
        
        **Min-Max (0-1):**
        - Roztáhne celé spektrum na rozsah 0-1
        - Může zvýraznit slabé signály
        
        ### Šablony
        
        - 💾 Uložte aktuální nastavení pro opakované použití
        - 📥 Exportujte všechny šablony do JSON souboru
        - 📤 Sdílejte šablony s kolegy importem JSON
        - 🔄 Rychlé přepínání mezi různými nastaveními
        
        ### Detekce píků
        
        **Automatická detekce:**
        - Používá algoritmus `scipy.signal.find_peaks`
        - Parametr "Citlivost" určuje minimální výšku píku
        - "Minimální vzdálenost" zabraňuje detekci duplicitních píků
        
        **Manuální úpravy:**
        - Přidat: zadejte pozice píků oddělené čárkou
        - Odstranit: zadejte pozice píků k odstranění
        - Aplikuje se vyhledávání lokálního maxima
        
        ### Tipy pro publikace
        
        1. **DPI:** Pro publikace používejte min. 300 DPI
        2. **Formát:** SVG je ideální pro další úpravy v Illustratoru
        3. **Barvy:** Pro černobílý tisk zvolte paletu "viridis" nebo "plasma"
        4. **Offset:** Nastavte tak, aby se spektra nepřekrývala, ale nebyla příliš daleko
        5. **Vyhlazení:** Zapněte pro hladší vzhled, ale ověřte, že nezkresluje data
        6. **Baseline:** Použijte ALS pro odstranění fluorescence před měřením píků
        7. **Normalizace:** Normalizujte před porovnáním intenzit mezi různými měřeními
        
        ### Pracovní postup
        
        1. Nahrajte soubory
        2. Aplikujte baseline korekci (pokud je třeba)
        3. Normalizujte spektra (pokud chcete srovnat intenzity)
        4. Vyhlaďte data
        5. Nastavte offset a rozsah
        6. Označte píky
        7. Uložte nastavení jako šablonu
        8. Exportujte graf
        """)
