# app.py — Instacart Reorder Prediction Dashboard


import os
import numpy as np
import pandas as pd
import joblib

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State

import plotly.graph_objects as go
import plotly.express as px


# -----------------------------
# CONFIG
# -----------------------------
TOP_USER_LIMIT = 5000
MAX_PRODUCTS_TO_SCORE = 1200
TOPN = 15

PROMO_TOPN = 15
PROMO_MIN_PROB = 0.40          # likely reorder
PROMO_GAP_QUANTILE = 0.75      # "long time" = top 25% for that user
PROMO_MIN_GAP_FLOOR = 3        # never require less than 3


# -----------------------------
# FILE CHECKS
# -----------------------------
def require_file(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")

require_file("rf_reorder_model.joblib")
require_file("rf_feature_columns.joblib")
require_file("user_features.csv")
require_file("product_features.csv")
require_file("user_product_features.csv")

HAS_LOOKUP = os.path.exists("products_lookup.csv")


# -----------------------------
# LOAD ARTIFACTS + DATA
# -----------------------------
model = joblib.load("rf_reorder_model.joblib")
feature_cols = joblib.load("rf_feature_columns.joblib")

user_features = pd.read_csv("user_features.csv")
product_features = pd.read_csv("product_features.csv")
user_product_features = pd.read_csv("user_product_features.csv")

# Types
user_features["user_id"] = user_features["user_id"].astype(int)
product_features["product_id"] = product_features["product_id"].astype(int)
user_product_features["user_id"] = user_product_features["user_id"].astype(int)
user_product_features["product_id"] = user_product_features["product_id"].astype(int)

# Optional product names
pid_to_name = {}
if HAS_LOOKUP:
    lookup = pd.read_csv("products_lookup.csv")
    if {"product_id", "product_name"}.issubset(lookup.columns):
        lookup["product_id"] = lookup["product_id"].astype(int)
        lookup["product_name"] = lookup["product_name"].astype(str)
        pid_to_name = dict(zip(lookup["product_id"], lookup["product_name"]))

def product_label(pid: int) -> str:
    name = pid_to_name.get(pid, "")
    if isinstance(name, str) and name.strip() and name.strip().lower() != "nan":
        return name
    return f"Product {pid}"

def prob_label(p: float) -> str:
    if p >= 0.70: return "High"
    if p >= 0.40: return "Medium"
    return "Low"

available_users = sorted(user_product_features["user_id"].unique().tolist())
if not available_users:
    raise ValueError("No users found in user_product_features.csv")

# Rank user-products by frequency (if available)
if "times_user_bought_product" in user_product_features.columns:
    upr_rank = user_product_features.sort_values(
        ["user_id", "times_user_bought_product"],
        ascending=[True, False]
    )
else:
    upr_rank = user_product_features.copy()
    upr_rank["times_user_bought_product"] = 0


# -----------------------------
# FEATURE ROW BUILDER
# -----------------------------
def build_feature_row(user_id: int, product_id: int) -> pd.DataFrame | None:
    u = user_features[user_features["user_id"] == user_id]
    p = product_features[product_features["product_id"] == product_id]
    up = user_product_features[
        (user_product_features["user_id"] == user_id) &
        (user_product_features["product_id"] == product_id)
    ]

    if u.empty or p.empty:
        return None

    if up.empty:
        up = pd.DataFrame([{
            "user_id": user_id,
            "product_id": product_id,
            "times_user_bought_product": 0,
            "user_product_reorder_rate": 0,
            "last_order_number_user_bought_product": 0,
            "first_order_number_user_bought_product": 0,
            "orders_since_last_purchase": 0
        }])

    row = pd.concat(
        [
            u.reset_index(drop=True),
            p.reset_index(drop=True),
            up.drop(columns=["user_id", "product_id"], errors="ignore").reset_index(drop=True)
        ],
        axis=1
    )
    row = row.reindex(columns=feature_cols, fill_value=0)
    return row


def score_user_products(user_id: int, max_products: int = MAX_PRODUCTS_TO_SCORE) -> pd.DataFrame:
    """Score this user's products and return a table with predicted probabilities."""
    sub = upr_rank[upr_rank["user_id"] == user_id].copy()
    if sub.empty:
        return pd.DataFrame()

    sub = sub.head(max_products)
    product_ids = sub["product_id"].astype(int).tolist()

    rows, kept = [], []
    for pid in product_ids:
        r = build_feature_row(user_id, pid)
        if r is not None:
            rows.append(r)
            kept.append(pid)

    if not rows:
        return pd.DataFrame()

    X = pd.concat(rows, axis=0)
    probs = model.predict_proba(X)[:, 1]

    out = sub[sub["product_id"].isin(kept)].copy()
    out["product_name"] = out["product_id"].apply(product_label)
    out["pred_reorder_prob"] = probs
    out["risk_band"] = out["pred_reorder_prob"].apply(prob_label)

    # Ensure expected columns exist for display + promo logic
    for c in ["orders_since_last_purchase", "user_product_reorder_rate"]:
        if c not in out.columns:
            out[c] = 0

    # Numeric safety
    out["orders_since_last_purchase"] = pd.to_numeric(out["orders_since_last_purchase"], errors="coerce").fillna(0)
    out["user_product_reorder_rate"] = pd.to_numeric(out["user_product_reorder_rate"], errors="coerce").fillna(0)
    out["pred_reorder_prob"] = pd.to_numeric(out["pred_reorder_prob"], errors="coerce").fillna(0)

    out = out.sort_values("pred_reorder_prob", ascending=False)
    return out


def empty_gauge():
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=0,
        number={"valueformat": ".3f"},
        gauge={"axis": {"range": [0, 1]}}
    ))
    fig.update_layout(margin=dict(l=20, r=20, t=10, b=10), height=260)
    return fig

def empty_bar():
    fig = px.bar(pd.DataFrame({"Product": [], "Probability": []}),
                 x="Probability", y="Product", orientation="h")
    fig.update_layout(margin=dict(l=20, r=20, t=10, b=10), height=420)
    return fig

def message_fig(msg: str):
    fig = go.Figure()
    fig.add_annotation(
        text=msg, x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper",
        font={"size": 14}
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=20, r=20, t=10, b=10), height=420)
    return fig


# -----------------------------
# DASH APP
# -----------------------------
app = dash.Dash(__name__)
app.title = "Instacart Reorder Prediction (Capstone)"

def kpi_box(title, value_id):
    return html.Div(
        style={
            "padding": "12px",
            "borderRadius": "14px",
            "background": "#f6f7fb",
            "boxShadow": "0 1px 6px rgba(0,0,0,0.08)",
            "minWidth": "190px",
        },
        children=[
            html.Div(title, style={"fontSize": "12px", "color": "#555"}),
            html.Div(id=value_id, style={"fontSize": "20px", "fontWeight": "800"})
        ]
    )

def card(title, children):
    return html.Div(
        style={"padding": "14px", "borderRadius": "14px", "background": "white",
               "boxShadow": "0 1px 8px rgba(0,0,0,0.10)"},
        children=[html.Div(title, style={"fontWeight": "800", "marginBottom": "10px"}), children]
    )

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "24px auto", "fontFamily": "Arial"},
    children=[
        html.H1("Instacart Reorder Prediction Dashboard"),
        html.Div(
            "Capstone demo: load a user, score their purchase history, and identify likely reorders and promotion candidates.",
            style={"color": "#555", "marginBottom": "18px"}
        ),

        html.Div(
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "alignItems": "end"},
            children=[
                html.Div(
                    style={"minWidth": "320px", "flex": "1"},
                    children=[
                        html.Label("Select a User", style={"fontWeight": "700"}),
                        dcc.Dropdown(
                            id="user_dropdown",
                            options=[{"label": str(u), "value": u} for u in available_users[:TOP_USER_LIMIT]],
                            placeholder="Search user_id",
                            searchable=True,
                            clearable=True
                        ),
                        html.Div(
                            f"Showing first {TOP_USER_LIMIT} users for performance.",
                            style={"fontSize": "12px", "color": "#777", "marginTop": "6px"}
                        ),
                    ],
                ),
                html.Button("Load User + Score Products", id="load_btn", n_clicks=0),
            ],
        ),

        html.Div(id="status_line", style={"marginTop": "10px", "color": "#444"}),
        html.Hr(style={"margin": "16px 0"}),

        html.Div(
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
            children=[
                kpi_box("User Total Orders", "kpi_total_orders"),
                kpi_box("User Reorder Ratio", "kpi_reorder_ratio"),
                kpi_box("Avg Days Between Orders", "kpi_avg_days"),
                kpi_box("Distinct Products Bought", "kpi_distinct_products"),
            ],
        ),

        html.Div(
            style={"display": "flex", "gap": "14px", "flexWrap": "wrap"},
            children=[
                html.Div(
                    style={"flex": "1.5", "minWidth": "640px"},
                    children=[
                        card(
                            "User Purchase History + Predicted Reorder Probability",
                            html.Div([
                                html.Div(
                                    "Tip: Use the table search box to find items by name. Click a row to view details.",
                                    style={"fontSize": "12px", "color": "#666", "marginBottom": "8px"}
                                ),
                                dash_table.DataTable(
                                    id="user_table",
                                    columns=[
                                        {"name": "Product", "id": "product_name"},
                                        {"name": "Product ID", "id": "product_id"},
                                        {"name": "Pred Prob", "id": "pred_reorder_prob", "type": "numeric",
                                         "format": {"specifier": ".3f"}},
                                        {"name": "Band", "id": "risk_band"},
                                        {"name": "Times Bought", "id": "times_user_bought_product"},
                                        {"name": "Orders Since Last Purchase", "id": "orders_since_last_purchase"},
                                        {"name": "User-Product Reorder Rate", "id": "user_product_reorder_rate"},
                                    ],
                                    data=[],
                                    page_size=12,
                                    filter_action="native",
                                    sort_action="native",
                                    row_selectable="single",
                                    selected_rows=[],
                                    style_table={"overflowX": "auto"},
                                    style_cell={"padding": "8px", "textAlign": "left", "fontSize": "13px"},
                                    style_header={"fontWeight": "700", "backgroundColor": "#f4f6fb"},
                                ),
                            ])
                        ),
                    ],
                ),

                html.Div(
                    style={"flex": "1", "minWidth": "420px"},
                    children=[
                        card(
                            "Selected Item Prediction",
                            html.Div([
                                html.Div(id="selected_text", style={"marginBottom": "10px", "fontSize": "14px"}),
                                dcc.Graph(id="gauge", figure=empty_gauge(), config={"displayModeBar": False}),
                                html.Div(id="explain_box", style={"fontSize": "13px", "color": "#444"}),
                            ])
                        ),
                    ],
                )
            ],
        ),

        html.Div(style={"height": "14px"}),

        html.Div(
            style={"display": "flex", "gap": "14px", "flexWrap": "wrap"},
            children=[
                html.Div(
                    style={"flex": "1", "minWidth": "520px"},
                    children=[
                        card(f"Top {TOPN} Reorder Recommendations",
                             dcc.Graph(id="topn_bar", figure=empty_bar(), config={"displayModeBar": False}))
                    ],
                ),
                html.Div(
                    style={"flex": "1", "minWidth": "520px"},
                    children=[
                        card(
                            "Promotion Candidates (Likely reorder + hasn’t purchased recently)",
                            dcc.Graph(id="promo_bar", figure=message_fig("Load a user to populate promotion candidates."),
                                      config={"displayModeBar": False})
                        )
                    ],
                ),
            ]
        ),
    ]
)


# -----------------------------
# CALLBACK 1: Load user + score + KPIs + charts
# -----------------------------
@app.callback(
    Output("user_table", "data"),
    Output("status_line", "children"),
    Output("topn_bar", "figure"),
    Output("promo_bar", "figure"),
    Output("kpi_total_orders", "children"),
    Output("kpi_reorder_ratio", "children"),
    Output("kpi_avg_days", "children"),
    Output("kpi_distinct_products", "children"),
    Input("load_btn", "n_clicks"),
    State("user_dropdown", "value"),
)
def load_user(n, user_id):
    if n is None or n == 0:
        return [], "Select a user and click “Load User + Score Products”.", empty_bar(), message_fig("Load a user to populate promotion candidates."), "—", "—", "—", "—"

    if user_id is None:
        return [], "No user selected.", empty_bar(), message_fig("No user selected."), "—", "—", "—", "—"

    user_id = int(user_id)

    # KPIs
    uf = user_features[user_features["user_id"] == user_id]
    if uf.empty:
        kpis = ("N/A", "N/A", "N/A", "N/A")
    else:
        r = uf.iloc[0]
        kpis = (
            str(int(r.get("user_total_orders", 0))),
            f"{float(r.get('user_reorder_ratio', 0)):.3f}",
            f"{float(r.get('user_avg_days_between', 0)):.2f}",
            str(int(r.get("user_distinct_products", 0))),
        )

    df = score_user_products(user_id)
    if df.empty:
        return [], f"No products could be scored for user {user_id}.", empty_bar(), message_fig("No promotion candidates available."), *kpis

    # Top-N chart
    topn = df.head(TOPN).copy()
    topn["label"] = topn["product_name"].astype(str)

    top_fig = px.bar(
        topn.sort_values("pred_reorder_prob", ascending=True),
        x="pred_reorder_prob",
        y="label",
        orientation="h",
        labels={"pred_reorder_prob": "Reorder Probability", "label": "Product"}
    )
    top_fig.update_layout(margin=dict(l=20, r=20, t=10, b=10), height=420)

    # -------------------------
    # Promotion Candidates (adaptive + fallback)
    # -------------------------
    df["orders_since_last_purchase"] = pd.to_numeric(df["orders_since_last_purchase"], errors="coerce").fillna(0)

    # Adaptive gap cutoff = user's 75th percentile (min floor)
    gap_cut = float(df["orders_since_last_purchase"].quantile(PROMO_GAP_QUANTILE))
    gap_cut = max(gap_cut, PROMO_MIN_GAP_FLOOR)

    promo = df[
        (df["pred_reorder_prob"] >= PROMO_MIN_PROB) &
        (df["orders_since_last_purchase"] >= gap_cut)
    ].copy()

    # If empty, fallback: just pick items with largest gaps (still useful for "nudge" logic)
    if promo.empty:
        promo = df.sort_values(
            ["orders_since_last_purchase", "pred_reorder_prob"],
            ascending=[False, False]
        ).head(PROMO_TOPN).copy()

        promo_title = f"Fallback promo view: largest purchase gaps (gap cutoff ≈ {gap_cut:.0f})"
    else:
        promo = promo.sort_values(
            ["pred_reorder_prob", "orders_since_last_purchase"],
            ascending=[False, False]
        ).head(PROMO_TOPN).copy()

        promo_title = f"Promo candidates: prob ≥ {PROMO_MIN_PROB}, gap ≥ {gap_cut:.0f}"

    promo["label"] = promo["product_name"].astype(str)

    promo_fig = px.bar(
        promo.sort_values("pred_reorder_prob", ascending=True),
        x="pred_reorder_prob",
        y="label",
        orientation="h",
        labels={"pred_reorder_prob": "Reorder Probability", "label": "Product"}
    )
    promo_fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=10),
        height=420,
        title=promo_title
    )

    msg = f"Loaded {len(df):,} user-products for user {user_id}."
    return df.to_dict("records"), msg, top_fig, promo_fig, *kpis


# -----------------------------
# CALLBACK 2: Row selection -> gauge + explanation
# -----------------------------
@app.callback(
    Output("selected_text", "children"),
    Output("gauge", "figure"),
    Output("explain_box", "children"),
    Input("user_table", "derived_virtual_data"),
    Input("user_table", "selected_rows"),
)
def on_row_selected(rows, selected_rows):
    if not rows or not selected_rows:
        return "Click a row in the table to view prediction details.", empty_gauge(), ""

    row = rows[selected_rows[0]]
    pid = int(row["product_id"])
    prob = float(row["pred_reorder_prob"])
    band = row.get("risk_band", prob_label(prob))

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        number={"valueformat": ".3f"},
        gauge={"axis": {"range": [0, 1]}, "bar": {"color": "#4c78a8"}}
    ))
    fig.update_layout(margin=dict(l=20, r=20, t=10, b=10), height=260)

    explain = html.Ul([
        html.Li(f"Risk band: {band}"),
        html.Li(f"Times bought historically: {int(row.get('times_user_bought_product', 0))}"),
        html.Li(f"Orders since last purchase: {float(row.get('orders_since_last_purchase', 0)):.0f}"),
        html.Li(f"User–product reorder rate: {float(row.get('user_product_reorder_rate', 0)):.3f}"),
    ])

    title = f"Selected: {row.get('product_name', product_label(pid))} (ID: {pid}) → Probability: {prob:.3f} ({band})"
    return title, fig, explain


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)