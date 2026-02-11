from __future__ import annotations

import base64
from typing import List, Dict

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback

from core.io import spectrum_from_upload
from core.processing import normalize_spectrum


app = Dash(__name__)
app.title = "SERS Plotter (Dash MVP)"


app.layout = html.Div(
    style={"fontFamily": "Arial", "padding": "16px"},
    children=[
        html.H2("🧪 SERS Plotter – Dash (MVP)"),

        html.Div(
            style={"display": "flex", "gap": "16px"},
            children=[
                # LEFT PANEL
                html.Div(
                    style={
                        "width": "320px",
                        "border": "1px solid #ddd",
                        "borderRadius": "12px",
                        "padding": "12px",
                    },
                    children=[
                        html.H4("1) Upload"),
                        dcc.Upload(
                            id="upload",
                            children=html.Div(
                                ["Přetáhni soubory sem nebo klikni (TXT, 2 sloupce)"]
                            ),
                            multiple=True,
                            style={
                                "width": "100%",
                                "height": "80px",
                                "lineHeight": "80px",
                                "borderWidth": "2px",
                                "borderStyle": "dashed",
                                "borderRadius": "10px",
                                "textAlign": "center",
                                "cursor": "pointer",
                            },
                        ),

                        html.Hr(),

                        html.H4("2) Normalizace"),
                        dcc.Dropdown(
                            id="norm_method",
                            options=[
                                {"label": "Vypnuto", "value": "none"},
                                {"label": "Maximum = 1", "value": "max"},
                                {"label": "Plocha = 1", "value": "area"},
                                {"label": "Min-Max (0–1)", "value": "minmax"},
                            ],
                            value="none",
                            clearable=False,
                        ),

                        html.Br(),
                        html.Label("Offset mezi spektry"),
                        dcc.Slider(
                            id="offset",
                            min=0,
                            max=5000,
                            step=50,
                            value=800,
                            marks={0: "0", 1000: "1000", 2000: "2000", 4000: "4000"},
                        ),

                        html.Div(
                            id="status",
                            style={"marginTop": "10px", "fontSize": "12px", "color": "#555"},
                        ),
                    ],
                ),

                # MAIN PANEL
                html.Div(
                    style={"flex": 1},
                    children=[
                        dcc.Graph(id="graph", style={"height": "720px"}),
                        dcc.Store(id="store_spectra", data=[]),
                    ],
                ),
            ],
        ),
    ],
)


def _parse_upload(contents: str, filename: str):
    header, b64 = contents.split(",", 1)
    file_bytes = base64.b64decode(b64)
    s = spectrum_from_upload(filename, file_bytes)
    return {
        "id": s.id,
        "filename": s.filename,
        "x": s.x.tolist(),
        "y": s.y.tolist(),
    }


@callback(
    Output("store_spectra", "data"),
    Output("status", "children"),
    Input("upload", "contents"),
    State("upload", "filename"),
    prevent_initial_call=True,
)
def on_upload(list_contents, list_filenames):
    if not list_contents:
        return [], "Žádné soubory."

    spectra = []
    errors = []

    for c, fn in zip(list_contents, list_filenames):
        try:
            spectra.append(_parse_upload(c, fn))
        except Exception as e:
            errors.append(f"{fn}: {e}")

    status = f"Načteno: {len(spectra)} souborů."
    if errors:
        status += "  Chyby: " + " | ".join(errors[:3])
        if len(errors) > 3:
            status += f" (+{len(errors)-3} dalších)"

    return spectra, status


@callback(
    Output("graph", "figure"),
    Input("store_spectra", "data"),
    Input("norm_method", "value"),
    Input("offset", "value"),
)
def render_graph(spectra_data: List[Dict], norm_method: str, offset: float):
    fig = go.Figure()

    if not spectra_data:
        fig.update_layout(
            template="plotly_white",
            title="Nahraj spektra pro zobrazení",
            xaxis_title="Ramanův posun (cm⁻¹)",
            yaxis_title="Intenzita (a.u.)",
        )
        return fig

    for i, s in enumerate(spectra_data):
        x = np.asarray(s["x"], dtype=float)
        y = np.asarray(s["y"], dtype=float)

        if norm_method != "none":
            y = normalize_spectrum(y, x=x, method=norm_method)

        y = y + i * float(offset)

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=s["filename"],
            )
        )

    fig.update_layout(
        template="plotly_white",
        title="SERS spektra (Dash MVP)",
        xaxis_title="Ramanův posun (cm⁻¹)",
        yaxis_title="Intenzita (a.u.)",
        hovermode="x unified",
    )
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
