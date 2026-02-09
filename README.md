# SERS-analysisGPT
# SERS Plotter v12.1 - Kompletní dokumentace

## 🎯 Přehled

SERS Plotter v12.1 je komplexní nástroj pro zpracování a vizualizaci Ramanových spekter s důrazem na publikační kvalitu. Aplikace běží na Streamlit a nabízí tři režimy práce pro maximální flexibilitu.

## 🚀 Instalace a spuštění

### Požadavky

```bash
pip install streamlit pandas matplotlib numpy scipy plotly
```

### Spuštění

```bash
streamlit run sers_plotter_v12.1.py
```

Aplikace se otevře v prohlížeči na adrese `http://localhost:8501`

## 📋 Hlavní funkce

### 1. **Tři režimy práce**

#### 📊 Spektra s napětím (série)
- Automatická detekce hodnot napětí z názvů souborů (např. `sample_-100mV.txt`)
- Filtrování dopředného/zpětného skenu
- Automatické řazení podle napětí
- Rychlý výběr po krocích (např. každých 100 mV)
- Přidávání záporných znamének

#### 📈 Obecná spektra (libovolná)
- Pro jakákoliv spektra bez napěťové série
- Flexibilní pojmenování (název souboru, číslo, vlastní text)
- Různé způsoby řazení (abeceda, pořadí nahrání, vlastní)

#### 🔧 Pokročilý režim
- Kombinuje všechny možnosti
- Maximální kontrola nad každým parametrem
- Vhodné pro zkušené uživatele

### 2. **💾 Šablony nastavení**

**Ukládání:**
- Klikněte na tlačítko "💾 Šablony" v horní liště
- Rozbalte "➕ Uložit aktuální nastavení"
- Zadejte název a stiskněte "💾 Uložit šablonu"

**Načítání:**
- V menu "💾 Šablony" vyberte šablonu
- Klikněte na "📥" pro načtení

**Export/Import:**
- Exportujte všechny šablony jako JSON soubor
- Sdílejte s kolegy nebo zálohujte
- Importujte ze souboru

### 3. **🔬 Baseline korekce**

Odstraní fluorescenční pozadí ze spekter.

#### ALS (Asymmetric Least Squares) - Doporučeno
- **Nejlepší pro:** Fluorescenční pozadí, složité křivky
- **λ (lambda):** Vyhlazení baseline
  - Nízké (100,000): Přesně sleduje minima
  - Střední (1,000,000): Optimální pro většinu případů
  - Vysoké (10,000,000): Velmi hladká křivka
- **p (asymetrie):** Jak moc se přizpůsobit minimům
  - 0.001: Velmi asymetrické, preferuje minima
  - 0.01: Standardní hodnota
  - 0.1: Symetričtější
- **Iterace:** 10 je obvykle dostačujících

#### Polynomiální
- **Nejlepší pro:** Jednoduché, plynulé pozadí
- **Stupeň:**
  - 2-3: Mírně zakřivené pozadí
  - 4-6: Složitější křivky

#### Rolling Ball
- **Nejlepší pro:** Jednoduché pozadí bez výrazných změn
- **Velikost okna:** Větší = hladší baseline

### 4. **📊 Normalizace spekter**

#### Maximum = 1
- Nejjednodušší metoda
- Všechna spektra mají stejnou maximální intenzitu
- **Použití:** Vizuální porovnání tvarů

#### Plocha = 1
- Normalizuje podle plochy pod křivkou
- Zachovává relativní intenzity
- **Použití:** Kvantitativní porovnání

#### Min-Max (0-1)
- Roztáhne celý rozsah na 0-1
- Zvýrazní i slabé signály
- **Použití:** Když chcete vidět detaily v celém spektru

**Škálování po normalizaci:**
- Násobitel pro lepší vizualizaci
- Doporučeno 1000-5000 pro typické SERS spektra

### 5. **📍 Detekce a správa píků**

#### Automatická detekce
- **Citlivost:** Minimální prominence píku (100-500 typicky)
- **Min. vzdálenost:** Minimální odstup mezi píky (20-50)

#### Manuální úpravy
- **Přidat píky:** Zadejte pozice oddělené čárkou (např. `1001, 1320, 1580`)
- **Odstranit píky:** Zadejte pozice k odstranění (např. `220, 450`)

#### Zobrazení píků
- Nejvyšší spektrum (default)
- Nejnižší spektrum
- Všechna spektra
- Konkrétní spektrum

### 6. **🎨 Vzhled a export**

#### Rozměry
- **Publikace:** 1200×1000 px @ 300 DPI
- **Prezentace:** 1920×1080 px @ 150 DPI
- **Poster:** 2400×1800 px @ 300 DPI
- **Vlastní:** Libovolné rozměry

#### Palety barev
- `jet`: Klasická duha (dobrá pro kontrasty)
- `viridis`: Perceptuálně uniformní (doporučeno pro publikace)
- `plasma`: Teplé barvy
- `coolwarm`: Modro-červená

#### Formáty exportu
- **SVG:** Vektorový, ideální pro Illustrator
- **PNG:** Rastrový, univerzální
- **PDF:** Pro tisk a archivaci

## 🔄 Pracovní postup

### Základní workflow

1. **Příprava dat**
   - Nahrajte .txt soubory
   - Zkontrolujte, že jsou správně načtená

2. **Zpracování**
   - Aplikujte baseline korekci (pokud je fluorescence)
   - Normalizujte (pokud chcete porovnat intenzity)
   - Vyhlaďte data (Savitzky-Golay)

3. **Vizualizace**
   - Nastavte rozsah X
   - Upravte offset mezi spektry
   - Vyberte barevnou paletu

4. **Anotace**
   - Označte píky (automaticky nebo manuálně)
   - Upravte pozice a rotaci popisků

5. **Export**
   - Uložte nastavení jako šablonu
   - Exportujte v požadovaných formátech

### Pokročilý workflow pro sérii měření

1. **První sada spekter**
   - Zpracujte a nastavte vše ručně
   - Uložte jako šablonu "SERS_protocol_1"

2. **Další sady**
   - Nahrajte nová spektra
   - Načtěte šablonu "SERS_protocol_1"
   - Malé úpravy podle potřeby
   - Export

## 📝 Formát vstupních souborů

### .txt soubory

```
300.5    1234.5
301.0    1256.8
301.5    1289.2
...
```

- Dva sloupce: X (Ramanův posun) a Y (intenzita)
- Oddělovač: mezera nebo tabulátor
- Žádná hlavička

### Pojmenování souborů pro napěťový režim

```
sample_0mV.txt
sample_-50mV.txt
sample_-100mV.txt
sample_-100mV_reverse.txt
```

- Hodnota napětí: `XXXmV` nebo `-XXXmV`
- Směr skenu: `reverse`, `zp`, nebo `back` pro zpětný sken

## 💡 Tipy a triky

### Pro publikace

1. **Vysoké DPI:** Vždy používejte 300 DPI nebo více
2. **SVG formát:** Zachovává vektorovou kvalitu pro další úpravy
3. **Konzistentní styling:** Použijte šablony pro jednotný vzhled
4. **Baseline korekce:** Vždy aplikujte před měřením píků
5. **Kontrola píků:** Zkontrolujte automaticky detekované píky manuálně

### Optimalizace baseline korekce

**Pokud baseline překračuje spektrum:**
- Snižte λ (lambda)
- Zvyšte p (asymetrii)

**Pokud baseline příliš kolísá:**
- Zvyšte λ (lambda)
- Snižte počet iterací

**Pokud baseline neodstraní pozadí:**
- Zkuste jinou metodu (ALS vs Polynom)
- Upravte parametry
- Zkontrolujte, že není problém v datech

### Normalizace

**Kdy normalizovat:**
- Porovnání různých měření
- Různé koncentrace analytů
- Různé podmínky měření

**Kdy NEnormalizovat:**
- Kvantitativní analýza absolutních intenzit
- Když je důležitá relativní intenzita mezi měřeními

### Detekce píků

**Mnoho falešných píků:**
- Zvyšte citlivost (prominence)
- Zvyšte minimální vzdálenost
- Vyhlaďte spektrum více

**Chybí důležité píky:**
- Snižte citlivost
- Přidejte manuálně
- Zkontrolujte baseline korekci

## 🐛 Řešení problémů

### Soubor se nenačetl
- Zkontrolujte formát (2 sloupce, správný oddělovač)
- Zkontrolujte, že nejsou speciální znaky v datech
- Otevřete soubor v textovém editoru

### Graf vypadá divně
- Zkontrolujte rozsah X
- Upravte offset
- Zkuste jinou paletu barev

### Baseline korekce nefunguje
- Zkuste jinou metodu
- Upravte parametry postupně
- Použijte náhled baseline pro kontrolu

### Píky jsou špatně umístěné
- Vypněte auto-detekci a přidejte manuálně
- Upravte parametry detekce
- Zkontrolujte, že je aplikováno vyhlazení

## 📊 Příklady použití

### Příklad 1: Napěťová série SERS

**Soubory:**
```
AgNPs_0mV.txt
AgNPs_-100mV.txt
AgNPs_-200mV.txt
...
AgNPs_-1000mV.txt
```

**Nastavení:**
1. Režim: "Spektra s napětím"
2. Filtr: "Dopředný"
3. Záporné znamínko: ✓
4. Krok: 100 mV
5. Baseline: ALS (λ=1000000, p=0.01)
6. Píky: Nejvyšší spektrum

### Příklad 2: Porovnání různých látek

**Soubory:**
```
rhodamine.txt
crystal_violet.txt
methylene_blue.txt
```

**Nastavení:**
1. Režim: "Obecná spektra"
2. Popisky: "Název souboru"
3. Normalizace: Maximum = 1, škála 1000
4. Baseline: ALS
5. Píky: Všechna spektra

### Příklad 3: Časová série

**Soubory:**
```
measurement_t0.txt
measurement_t1.txt
measurement_t2.txt
...
```

**Nastavení:**
1. Režim: "Obecná spektra"
2. Popisky: "Vlastní" → "0 min", "5 min", "10 min"
3. Baseline: Polynom (stupeň 3)
4. Vyhlazení: ✓

## 🔗 Podpora a kontakt

Pro otázky, chyby nebo návrhy na vylepšení:
- GitHub Issues
- Email: [vaše_email@example.com]

## 📜 Změny verzí

### v12.1 (aktuální)
- ✅ Přidány šablony nastavení
- ✅ Baseline korekce (ALS, Polynom, Rolling Ball)
- ✅ Normalizace spekter
- ✅ Vylepšená dokumentace

### v12.0
- ✅ Tři režimy práce
- ✅ Pokročilá správa píků
- ✅ Export do více formátů
- ✅ Interaktivní náhled

### v11.0
- ✅ Základní funkcionalita
- ✅ Napěťový režim
- ✅ Automatická detekce píků

---

**Vytvořeno pro vědeckou komunitu | Streamlit aplikace pro zpracování SERS spekter**
