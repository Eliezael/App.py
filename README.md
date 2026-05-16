import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== PROTEÇÃO POR SENHA ====================
SENHA = "Esm.2026"

if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

if not st.session_state.autenticado:
    st.set_page_config(page_title="Acesso Restrito", layout="centered")
    st.title("🔒 Acesso Restrito")
    senha_input = st.text_input("Digite a senha para acessar:", type="password")
    if senha_input == SENHA:
        st.session_state.autenticado = True
        st.rerun()
    else:
        if senha_input:
            st.error("Senha incorreta!")
        st.stop()
# ============================================================

st.set_page_config(page_title="REIT AI v23.0", layout="wide")
st.title("🏢 REIT AI v23.0 - Macro & Portfolio")

tickers = ["VICI", "PLD", "O", "DOC", "SBRA", "AMT", "EQIX"]
selected = st.multiselect("REITs para carteira", tickers, default=tickers[:3])

# Abas
tabs = st.tabs(["📊 Dados", "📈 Otimização", "📉 Risco"])

with tabs[0]:
    st.subheader("Dados fundamentais")
    for t in selected:
        info = yf.Ticker(t).info
        st.write(f"**{t}** - Preço: ${info.get('currentPrice',0):.2f} | DY: {info.get('dividendYield',0)*100:.2f}% | P/FFO: {info.get('trailingPE',0):.1f}")

with tabs[1]:
    st.subheader("Otimização de carteira (Markowitz)")
    if len(selected) >= 2:
        # dados de retorno
        rets = {}
        for t in selected:
            df = yf.Ticker(t).history(period="2y")
            if not df.empty:
                rets[t] = df['Close'].pct_change().dropna()
        if rets:
            df_rets = pd.DataFrame(rets).dropna()
            mean_returns = df_rets.mean() * 252
            cov = df_rets.cov() * 252
            # simulação simples
            num = 5000
            results = np.zeros((3, num))
            weights_record = []
            for i in range(num):
                w = np.random.random(len(selected))
                w /= w.sum()
                weights_record.append(w)
                ret = np.sum(mean_returns * w)
                risk = np.sqrt(np.dot(w.T, np.dot(cov, w)))
                sharpe = ret / risk
                results[0,i] = ret
                results[1,i] = risk
                results[2,i] = sharpe
            best_idx = np.argmax(results[2])
            best_weights = weights_record[best_idx]
            st.write("**Pesos ótimos (máximo Sharpe):**")
            for t, w in zip(selected, best_weights):
                st.write(f"{t}: {w:.2%}")
            st.metric("Retorno esperado", f"{results[0,best_idx]:.2%}")
            st.metric("Risco esperado", f"{results[1,best_idx]:.2%}")
            st.metric("Sharpe Ratio", f"{results[2,best_idx]:.2f}")
    else:
        st.info("Selecione pelo menos 2 REITs.")

with tabs[2]:
    st.subheader("Value at Risk (VaR)")
    risk_ticker = st.selectbox("REIT", selected, key="risk")
    df = yf.Ticker(risk_ticker).history(period="1y")
    if not df.empty:
        rets = df['Close'].pct_change().dropna()
        var_95 = np.percentile(rets, 5)
        cvar_95 = rets[rets <= var_95].mean()
        st.metric("VaR 95%", f"{var_95:.2%}")
        st.metric("CVaR 95%", f"{cvar_95:.2%}")
