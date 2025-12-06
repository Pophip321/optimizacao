import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
import firebase_admin
from firebase_admin import credentials, auth, firestore
import random
import time
from datetime import datetime


# ========== FIREBASE INIT ==========
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

# ========== SESSION STATE ==========
if "logado" not in st.session_state:
    st.session_state.logado = False
if "email" not in st.session_state:
    st.session_state.email = ""
if "codigo" not in st.session_state:
    st.session_state.codigo = ""
if "permanecer" not in st.session_state:
    st.session_state.permanecer = False


def tela_login():
    st.title("🔐 Acesso ao Sistema")

    email = st.text_input("Digite seu e-mail")

    permanecer = st.checkbox("Permanecer logado neste dispositivo")
    
    if st.button("Enviar código"):
        if email:
            codigo = str(random.randint(100000, 999999))
            st.session_state.codigo = codigo
            st.session_state.email = email
            st.session_state.permanecer = permanecer

            # salva no firebase
            db.collection("logins").add({
                "email": email,
                "codigo": codigo,
                "criado_em": datetime.now()
            })

            st.success("Código enviado para seu e-mail (simulação).")
            st.info(f"Código gerado (debug): {codigo}")
        else:
            st.error("Digite um e-mail válido")

    codigo_digitado = st.text_input("Digite o código recebido")

    if st.button("Confirmar acesso"):
        if codigo_digitado == st.session_state.codigo:
            st.session_state.logado = True

            db.collection("acessos").add({
                "email": st.session_state.email,
                "hora_acesso": datetime.now()
            })

            st.rerun()
        else:
            st.error("Código incorreto")

if not st.session_state.logado:
    tela_login()
    st.stop()




st.set_page_config(page_title="Otimização de Portfólio", page_icon="📊", layout="wide")

st.title("📊 Otimização de Portfólio de Ações")
st.markdown("Análise de fronteira eficiente e estratégias de alocação de ativos")

with st.sidebar:
    st.header("⚙️ Configurações")
    
    st.subheader("1️⃣ Período de Análise")
    col1, col2 = st.columns(2)
    with col1:
        data_inicio = st.date_input(
            "Data Inicial",
            value=dt.date(2020, 12, 1),
            max_value=dt.date.today()
        )
    with col2:
        data_fim = st.date_input(
            "Data Final",
            value=dt.date.today(),
            max_value=dt.date.today()
        )
    
    st.subheader("2️⃣ Ativos (Tickers)")
    st.markdown("Digite os tickers (ex: PETR4, VALE3, BOVA11)")
    
    ativos_input = st.text_area(
        "Um ticker por linha, como já esta o exemplo:",
        value="PETR4\nVALE3\nITUB4\nBBDC4\nABEV3",
        height=150
    )
    
    st.subheader("3️⃣ Opções")
    usar_ajustada = st.checkbox(
        "Usar cotação ajustada (dividendos/splits)",
        value=True,
        help="Considera proventos, bonificações e splits"
    )
    
    usar_log = st.checkbox(
        "Usar retornos logarítmicos",
        value=False,
        help="Caso contrário, usa retornos simples"
    )
    
    st.subheader("4️⃣ Limite Máximo por Ativo (%)")
    limite_peso = st.slider(
        "Escolha o limite máximo permitido para uma carteira de pesos restritos",
        min_value=0,
        max_value=100,
        value=0,
        step=1,
        help="0% = desativa carteiras limitadas | 1-100% = peso máximo por ativo"
    ) / 100



    target_return = st.number_input(
        "Retorno alvo mensal (%)",
        min_value=-10.0,
        max_value=20.0,
        value=1.0,
        step=0.1,
        help="Para estratégia Target"
    ) / 100
    
    st.subheader("5️⃣ Carteira Personalizada (Opcional)")
    analisar_carteira = st.checkbox("Analisar minha carteira", value=False)
    
    carteira_input = ""
    if analisar_carteira:
        carteira_input = st.text_area(
            "Carteira (TICKER PESO%)",
            value="PETR4 25\nVALE3 25\nITUB4 50",
            height=100,
            help="Digite um ativo e seu peso por linha. Ex: PETR4 25"
        )
    
    calcular = st.button("🚀 Calcular Portfólios", type="primary", use_container_width=True)

if calcular:
    try:
        with st.spinner("Processando dados..."):
            # VALIDAÇÃO ORIGINAL DOS ATIVOS (como na primeira versão)
            inicio_original = dt.datetime.combine(data_inicio, dt.datetime.min.time())
            fim_original = dt.datetime.combine(data_fim, dt.datetime.min.time())
            
            ativos_digitados = [linha.strip().upper() for linha in ativos_input.split('\n') if linha.strip()]
            ativos_digitados = list(dict.fromkeys(ativos_digitados))
            
            if not ativos_digitados:
                st.error("Por favor, insira pelo menos um ativo.")
                st.stop()
            
            st.info(f"Validando {len(ativos_digitados)} ativos no Yahoo Finance...")
            
            ativos_ok = []
            ativos_errados = []
            
            progress_bar = st.progress(0)
            for idx, ticker in enumerate(ativos_digitados):
                yf_ticker = ticker + ".SA"
                t = yf.Ticker(yf_ticker)
                try:
                    info = t.history(period="1d")
                    if info is None or info.empty:
                        ativos_errados.append(ticker)
                    else:
                        ativos_ok.append(yf_ticker)
                except:
                    ativos_errados.append(ticker)
                progress_bar.progress((idx + 1) / len(ativos_digitados))
            
            progress_bar.empty()
            
            if ativos_errados:
                st.warning(f"⚠️ Ativos não encontrados: {', '.join(ativos_errados)}")
            
            if not ativos_ok:
                st.error("Nenhum ativo válido encontrado!")
                st.stop()
            
            lista_acoes = ativos_ok
            st.success(f"✅ {len(lista_acoes)} ativos válidos: {', '.join([a.replace('.SA', '') for a in lista_acoes])}")
            
            with st.spinner("Baixando dados históricos..."):
                # =============================================================
                # AJUSTE AUTOMÁTICO DAS DATAS PARA GARANTIR MESES COMPLETOS
                # =============================================================
                
                def ajustar_data_mes(data, eh_final=False):
                    """Ajusta a data para a última cotação disponível do mês"""
                    hoje = dt.date.today()
                    
                    # Se é data final e é mês atual, usa ontem
                    if eh_final and data.year == hoje.year and data.month == hoje.month:
                        return hoje - dt.timedelta(days=1)
                    
                    # Para outros casos, busca a última cotação disponível do mês
                    try:
                        # Testa com o primeiro ativo válido
                        ticker_teste = lista_acoes[0]
                        t = yf.Ticker(ticker_teste)
                        
                        # Define período de busca (todo o mês da data)
                        inicio_mes = dt.datetime(data.year, data.month, 1)
                        if data.month == 12:
                            fim_mes = dt.datetime(data.year + 1, 1, 1)
                        else:
                            fim_mes = dt.datetime(data.year, data.month + 1, 1)
                        
                        hist = t.history(start=inicio_mes, end=fim_mes)
                        if not hist.empty:
                            return hist.index[-1].date()
                        
                    except:
                        pass
                    
                    # Fallback: se não conseguir, usa o último dia útil provável do mês
                    return data.replace(day=1) + dt.timedelta(days=32)
                    data_ajustada = data_ajustada.replace(day=1) - dt.timedelta(days=1)
                    return data_ajustada
                
                # Aplica os ajustes
                data_inicio_ajustada = ajustar_data_mes(data_inicio, eh_final=False)
                data_fim_ajustada = ajustar_data_mes(data_fim, eh_final=True)
                
                # Mostra ajuste se necessário
                if data_inicio_ajustada != data_inicio or data_fim_ajustada != data_fim:
                    st.info(f"🔍 **Ajuste automático para cotações disponíveis:**\n"
                        f"- Data inicial: {data_inicio} → {data_inicio_ajustada}\n"
                        f"- Data final: {data_fim} → {data_fim_ajustada}")
                
                # Usa as datas ajustadas
                inicio = dt.datetime.combine(data_inicio_ajustada, dt.datetime.min.time())
                fim = dt.datetime.combine(data_fim_ajustada, dt.datetime.min.time())
                
                # Baixa dados com datas ajustadas
                dados = yf.download(lista_acoes, start=inicio, end=fim, auto_adjust=usar_ajustada, progress=False)
                
                # Resto do código original...
                if usar_ajustada:
                    if hasattr(dados.columns, "levels") and "Adj Close" in dados.columns.levels[0]:
                        precos = dados["Adj Close"].copy()
                    else:
                        precos = dados["Close"].copy()
                else:
                    precos = dados["Close"].copy()
                
                if isinstance(precos.columns, pd.MultiIndex):
                    precos.columns = precos.columns.get_level_values(0)
                
                # Pega a última cotação de cada mês (já garantido pelos ajustes)
                precos_mensais = precos.resample("ME").last()
                
                # Remove meses onde não há dados suficientes
                meses_por_ativo = precos_mensais.count()
                ativos_validos_24m = meses_por_ativo[meses_por_ativo >= 24].index
                precos_mensais = precos_mensais[ativos_validos_24m]
                
                if len(ativos_validos_24m) == 0:
                    st.error("Nenhum ativo possui ao menos 24 meses de histórico. Ajuste a janela temporal.")
                    st.stop()
                
                # SALVA O NÚMERO DE MESES PARA USAR NO TAB4
                numero_meses_efetivo = len(precos_mensais)
                
                if usar_log:
                    retornos = np.log(precos_mensais / precos_mensais.shift(1))
                else:
                    retornos = precos_mensais.pct_change()
                
                retornos = retornos.dropna(how="all")
                retornos = retornos.dropna(axis=1, how="all")
                
                ativos_validos = list(retornos.columns)
                n = len(ativos_validos)
                
                if n == 0:
                    st.error("Nenhum ativo com retornos válidos após cálculo.")
                    st.stop()
            
            with st.spinner("Calculando métricas..."):
                media_retorno = retornos.mean()
                matriz_covariancia = retornos.cov()
                matriz_corr = retornos.corr()
                matriz_corr_vals = matriz_corr.values
                vol_individual = np.sqrt(np.diag(matriz_covariancia.values))
                
                def risco_portfolio(w):
                    w = np.array(w)
                    return float(np.sqrt(w.T @ matriz_covariancia.loc[ativos_validos, ativos_validos].values @ w))
                
                def retorno_portfolio(w):
                    w = np.array(w)
                    return float(np.dot(w, media_retorno[ativos_validos]))
                
                def sharpe_negativo(w):
                    vol = risco_portfolio(w)
                    if vol <= 0:
                        return 1e6
                    return - retorno_portfolio(w) / vol
                
                def soma_pesos(w):
                    return np.sum(w) - 1.0
                
                def correlacao_media(w):
                    w = np.array(w)
                    return float(np.sum(np.outer(w, w) * matriz_corr_vals))
                
                bounds = tuple((0.0, 1.0) for _ in range(len(ativos_validos)))
                w0 = np.ones(len(ativos_validos)) / len(ativos_validos)
                
                cov_matrix = matriz_covariancia.loc[ativos_validos, ativos_validos].values
                mean_returns = media_retorno[ativos_validos].values
            
            # Monte Carlo removido para otimização
            # with st.spinner(f"Simulando {numero_carteiras:,} carteiras (Monte Carlo)..."):
            #     np.random.seed(42)
            #     T_tabela_pesos = np.random.random((numero_carteiras, n))
            #     T_tabela_pesos = T_tabela_pesos / T_tabela_pesos.sum(axis=1, keepdims=True)
            #     
            #     tabela_retornos = T_tabela_pesos @ mean_returns
            #     variances = np.sum(T_tabela_pesos @ cov_matrix * T_tabela_pesos, axis=1)
            #     tabela_volatilidades = np.sqrt(variances)
            #     tabela_sharpe = np.divide(tabela_retornos, tabela_volatilidades, out=np.full_like(tabela_retornos, -np.inf), where=tabela_volatilidades!=0)
            
            with st.spinner("Otimizando estratégias..."):

                # ------------------------------------------------------------
                # 1) Carteira Máximo Sharpe (sem limite)
                # ------------------------------------------------------------
                res_maxsharpe = minimize(
                    sharpe_negativo, w0, method="SLSQP", bounds=bounds,
                    constraints={'type': 'eq', 'fun': soma_pesos}
                )
                pesos_max_sharpe = res_maxsharpe.x if res_maxsharpe.success else w0.copy()
                ret_max_sharpe = retorno_portfolio(pesos_max_sharpe)
                vol_max_sharpe = risco_portfolio(pesos_max_sharpe)
                sharpe_max_sharpe = ret_max_sharpe / vol_max_sharpe


                # ------------------------------------------------------------
                # 2) Carteira Mínima Volatilidade (sem limite)
                # ------------------------------------------------------------
                res_minvar = minimize(
                    risco_portfolio, w0, method="SLSQP", bounds=bounds,
                    constraints={'type': 'eq', 'fun': soma_pesos}
                )
                pesos_min_vol = res_minvar.x if res_minvar.success else w0.copy()
                ret_min_vol = retorno_portfolio(pesos_min_vol)
                vol_min_vol = risco_portfolio(pesos_min_vol)
                sharpe_min_vol = ret_min_vol / vol_min_vol


                # ------------------------------------------------------------
                # 3) Target Return
                # ------------------------------------------------------------
                ret_min = float(media_retorno[ativos_validos].min())
                ret_max = float(media_retorno[ativos_validos].max())

                target_return = np.clip(target_return, ret_min, ret_max)

                def constraint_retorno(w):
                    return np.dot(w, media_retorno[ativos_validos]) - target_return

                res_target = minimize(
                    risco_portfolio, w0, method="SLSQP", bounds=bounds,
                    constraints=[
                        {'type': 'eq', 'fun': soma_pesos},
                        {'type': 'eq', 'fun': constraint_retorno}
                    ]
                )

                w_target = res_target.x if res_target.success else w0.copy()
                ret_target = retorno_portfolio(w_target)
                vol_target = risco_portfolio(w_target)
                sharpe_target = ret_target / vol_target


                # ------------------------------------------------------------
                # 4) Fronteira eficiente
                # ------------------------------------------------------------
                alvo_rets = np.linspace(ret_min, ret_max, 80)
                fronteira_rets = []
                fronteira_vols = []

                for r in alvo_rets:
                    cons = [
                        {'type': 'eq', 'fun': soma_pesos},
                        {'type': 'eq', 'fun': lambda w, r=r: np.dot(w, media_retorno[ativos_validos]) - r}
                    ]
                    res = minimize(risco_portfolio, w0, method="SLSQP", bounds=bounds, constraints=cons)
                    if res.success:
                        fronteira_rets.append(np.dot(res.x, media_retorno[ativos_validos]))
                        fronteira_vols.append(risco_portfolio(res.x))


                # =============================================================
                # 5) OTIMIZAÇÕES COM LIMITE DE PESO (SÓ SE limite_peso > 0)
                # =============================================================

                # Inicializar variáveis com None
                pesos_max_sharpe_lim = None
                ret_max_sharpe_lim = None
                vol_max_sharpe_lim = None
                sharpe_max_sharpe_lim = None
                pesos_min_vol_lim = None
                ret_min_vol_lim = None
                vol_min_vol_lim = None
                sharpe_min_vol_lim = None

                if limite_peso > 0:
                    limite_necessario = 1 / n
                    if limite_peso < limite_necessario:
                        limite_peso = limite_necessario

                    bounds_limitados = tuple((0.0, limite_peso) for _ in range(n))

                    # --- Máx Sharpe limitado
                    try:
                        res_maxsharpe_lim = minimize(
                            sharpe_negativo,
                            w0,
                            method="SLSQP",
                            bounds=bounds_limitados,
                            constraints={'type': 'eq', 'fun': soma_pesos}
                        )
                        if res_maxsharpe_lim.success:
                            pesos_max_sharpe_lim = res_maxsharpe_lim.x
                            ret_max_sharpe_lim = retorno_portfolio(pesos_max_sharpe_lim)
                            vol_max_sharpe_lim = risco_portfolio(pesos_max_sharpe_lim)
                            sharpe_max_sharpe_lim = ret_max_sharpe_lim / vol_max_sharpe_lim if vol_max_sharpe_lim != 0 else 0
                    except:
                        pass

                    # --- Min Vol limitado
                    try:
                        res_minvol_lim = minimize(
                            risco_portfolio,
                            w0,
                            method="SLSQP",
                            bounds=bounds_limitados,
                            constraints={'type': 'eq', 'fun': soma_pesos}
                        )
                        if res_minvol_lim.success:
                            pesos_min_vol_lim = res_minvol_lim.x
                            ret_min_vol_lim = retorno_portfolio(pesos_min_vol_lim)
                            vol_min_vol_lim = risco_portfolio(pesos_min_vol_lim)
                            sharpe_min_vol_lim = ret_min_vol_lim / vol_min_vol_lim if vol_min_vol_lim != 0 else 0
                    except:
                        pass



            pesos_user = None
            ret_user = vol_user = sharpe_user = corr_user = None
            
            if analisar_carteira and carteira_input.strip():
                with st.spinner("Analisando sua carteira..."):
                    carteira_raw = {}
                    for linha in carteira_input.strip().split('\n'):
                        linha = linha.strip()
                        if not linha:
                            continue
                        partes = linha.split()
                        if len(partes) != 2:
                            continue
                        ticker, peso_str = partes
                        try:
                            peso = float(peso_str.replace(",", "."))
                            ticker_full = ticker.upper()
                            if not ticker_full.endswith(".SA"):
                                ticker_full += ".SA"
                            carteira_raw[ticker_full] = peso
                        except:
                            continue
                    
                    if carteira_raw:
                        pesos_user = np.zeros(len(ativos_validos))
                        for i, ativo in enumerate(ativos_validos):
                            if ativo in carteira_raw:
                                pesos_user[i] = carteira_raw[ativo]
                        soma = pesos_user.sum()
                        if soma > 0:
                            pesos_user = pesos_user / soma
                            ret_user = retorno_portfolio(pesos_user)
                            vol_user = risco_portfolio(pesos_user)
                            sharpe_user = ret_user / vol_user if vol_user != 0 else np.nan
                            corr_user = correlacao_media(pesos_user)
        
        st.success("✅ Cálculos concluídos!")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Fronteira Eficiente", "📊 Estratégias", "💼 Ativos Individuais", "📋 Detalhes"])
        
        with tab1:
            st.subheader("Fronteira Eficiente e Estratégias de Alocação")
            
            fig = go.Figure()
            
            # =============================================================
            # 1) FRONTEIRA EFICIENTE
            # =============================================================
            if len(fronteira_vols) > 0:
                fig.add_trace(go.Scatter(
                    x=np.array(fronteira_vols) * 100,  # Converte volatilidade para %
                    y=np.array(fronteira_rets) * 100,  # Converte retorno para %
                    mode='lines',
                    line=dict(color='orange', width=3),
                    name='Fronteira Eficiente'
                ))
            
            # =============================================================
            # 2) ESTRATÉGIAS PRINCIPAIS (SEMPRE EXISTEM)
            # =============================================================
            estrategias = [
                (vol_max_sharpe * 100, ret_max_sharpe * 100, 'Máx Sharpe', 'red', 12),
                (vol_min_vol * 100, ret_min_vol * 100, 'Mín Risco', 'blue', 12),
                (vol_target * 100, ret_target * 100, 'Target', 'white', 12),
            ]
            
            # Adiciona as estratégias principais ao gráfico
            for vol, ret, nome, cor, tamanho in estrategias:
                fig.add_trace(go.Scatter(
                    x=[vol],
                    y=[ret],
                    mode='markers',
                    marker=dict(size=tamanho, color=cor),
                    name=nome,
                    hovertemplate=f'{nome}<br>Vol: %{{x:.2f}}%<br>Ret: %{{y:.2f}}%<extra></extra>'
                ))
            
            # =============================================================
            # 3) ESTRATÉGIAS LIMITADAS (SÓ SE limite_peso > 0)
            # =============================================================
            if limite_peso > 0:
                estrategias_limitadas = [
                    (vol_max_sharpe_lim * 100, ret_max_sharpe_lim * 100, 'Sharpe Limitado', 'green', 'cross'),
                    (vol_min_vol_lim   * 100, ret_min_vol_lim   * 100, 'MinVol Limitado', 'purple',  'cross')
                ]

                for vol, ret, nome, cor, simbolo in estrategias_limitadas:
                    fig.add_trace(go.Scatter(
                        x=[vol],
                        y=[ret],
                        mode='markers',
                        marker=dict(
                            size=14,
                            color=cor,
                            symbol=simbolo,  # Usa símbolo de cruz para diferenciar
                            line=dict(color='black', width=2)
                        ),
                        name=nome,
                        hovertemplate=f'{nome}<br>Vol: %{{x:.2f}}%<br>Ret: %{{y:.2f}}%<extra></extra>'
                    ))
            
            # =============================================================
            # 4) CARTEIRA DO USUÁRIO (SE EXISTIR)
            # =============================================================
            if pesos_user is not None:
                fig.add_trace(go.Scatter(
                    x=[vol_user * 100],
                    y=[ret_user * 100],
                    mode='markers',
                    marker=dict(
                        size=15, 
                        color='white', 
                        line=dict(color='black', width=2), 
                        symbol='star'  # Símbolo de estrela para destacar
                    ),
                    name='Minha Carteira',
                    hovertemplate='Minha Carteira<br>Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<extra></extra>'
                ))
            
            # =============================================================
            # 5) CONFIGURAÇÕES DO GRÁFICO
            # =============================================================
            fig.update_layout(
                xaxis_title="Volatilidade Mensal (%)",
                yaxis_title="Retorno Mensal (%)",
                hovermode='closest',  # Mostra tooltip do ponto mais próximo
                height=600,
                showlegend=True,
                legend=dict(x=0.01, y=0.99),  # Legenda no canto superior esquerdo
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Comparação de Estratégias")
            
            # =============================================================
            # CALCULAR RETORNOS ACUMULADOS PARA TODAS AS ESTRATÉGIAS
            # =============================================================
            def calcular_retornos_acumulados(pesos):
                """Calcula ambos retornos: acumulado real e esperado"""
                if pesos is None:
                    return None, None
                retornos_carteira = retornos.dot(pesos)
                
                # Retorno acumulado REAL (o que realmente aconteceu)
                retorno_acumulado_real = (1 + retornos_carteira).prod() - 1
                
                # Retorno acumulado ESPERADO (projeção com média - juros compostos)
                retorno_medio_mensal = retornos_carteira.mean()
                n_meses = len(retornos_carteira)
                retorno_acumulado_esperado = (1 + retorno_medio_mensal) ** n_meses - 1
                
                return retorno_acumulado_real, retorno_acumulado_esperado

            # Calcular retornos acumulados para cada estratégia
            ret_acum_real_max_sharpe, ret_acum_esperado_max_sharpe = calcular_retornos_acumulados(pesos_max_sharpe)
            ret_acum_real_min_vol, ret_acum_esperado_min_vol = calcular_retornos_acumulados(pesos_min_vol)
            ret_acum_real_target, ret_acum_esperado_target = calcular_retornos_acumulados(w_target)

            # Só calcular para carteiras limitadas se existirem
            if limite_peso > 0:
                ret_acum_real_max_sharpe_lim, ret_acum_esperado_max_sharpe_lim = calcular_retornos_acumulados(pesos_max_sharpe_lim)
                ret_acum_real_min_vol_lim, ret_acum_esperado_min_vol_lim = calcular_retornos_acumulados(pesos_min_vol_lim)
            else:
                ret_acum_real_max_sharpe_lim = ret_acum_esperado_max_sharpe_lim = None
                ret_acum_real_min_vol_lim = ret_acum_esperado_min_vol_lim = None

            if pesos_user is not None:
                ret_acum_real_user, ret_acum_esperado_user = calcular_retornos_acumulados(pesos_user)

            if pesos_user is not None:
                ret_acum_user = calcular_retornos_acumulados(pesos_user)
            
            # =============================================================
            # 1) TABELA RESUMO DAS ESTRATÉGIAS
            # =============================================================
            estrategias_data = {
                'Estratégia': [
                    'Máximo Sharpe',
                    'Mínima Volatilidade',
                    'Target'
                ],
                'Retorno Mensal (%)': [
                    ret_max_sharpe * 100,
                    ret_min_vol * 100,
                    ret_target * 100
                ],
                'Retorno Acumulado Real (%)': [
                    ret_acum_real_max_sharpe * 100,
                    ret_acum_real_min_vol * 100,
                    ret_acum_real_target * 100
                ],
                'Volatilidade (%)': [
                    vol_max_sharpe * 100,
                    vol_min_vol * 100,
                    vol_target * 100
                ],
                'Sharpe Ratio': [  
                    sharpe_max_sharpe,
                    sharpe_min_vol,
                    sharpe_target
                ]
            }

            # Adiciona carteiras limitadas apenas se existirem
            if limite_peso > 0 and ret_acum_real_max_sharpe_lim is not None and ret_acum_real_min_vol_lim is not None:
                estrategias_data['Estratégia'].extend(['Sharpe Limitado', 'Vol Min Limitada'])
                estrategias_data['Retorno Mensal (%)'].extend([ret_max_sharpe_lim * 100, ret_min_vol_lim * 100])
                estrategias_data['Volatilidade (%)'].extend([vol_max_sharpe_lim * 100, vol_min_vol_lim * 100])
                estrategias_data['Sharpe Ratio'].extend([sharpe_max_sharpe_lim, sharpe_min_vol_lim])  
                estrategias_data['Retorno Acumulado Real (%)'].extend([ret_acum_real_max_sharpe_lim * 100, ret_acum_real_min_vol_lim *100])

            # Adiciona carteira do usuário
            if pesos_user is not None:
                estrategias_data['Estratégia'].append('🌟 Minha Carteira')
                estrategias_data['Retorno Mensal (%)'].append(ret_user * 100)
                estrategias_data['Volatilidade (%)'].append(vol_user * 100)
                estrategias_data['Sharpe Ratio'].append(sharpe_user)  
                estrategias_data['Retorno Acumulado Real (%)'].append(ret_acum_real_user * 100)
                
                        
            df_estrategias = pd.DataFrame(estrategias_data)
            df_estrategias = df_estrategias.sort_values('Sharpe Ratio', ascending=False)
            
            st.dataframe(
                df_estrategias.style.format({
                    'Retorno Mensal (%)': '{:.4f}',
                    'Volatilidade (%)': '{:.4f}',
                    'Sharpe Ratio': '{:.4f}',  
                    'Retorno Acumulado Real (%)': '{:.4f}'
                }).background_gradient(subset=['Sharpe Ratio'], cmap='RdYlGn'),  
                use_container_width=True,
                hide_index=True
            )
            
            st.markdown("---")
            
            # =============================================================
            # 2) COLUNAS PARA AS PRINCIPAIS CARTEIRAS
            # =============================================================
            col1, col2 = st.columns(2)
            
            # -------------------------------------------------------------
            # 🏆 Máximo Sharpe
            # -------------------------------------------------------------
            with col1:
                st.markdown("### 🏆 Máximo Sharpe")
                df_max_sharpe = pd.DataFrame({
                    'Ativo': [a.replace('.SA', '') for a in ativos_validos],
                    'Peso (%)': pesos_max_sharpe * 100
                })
                df_max_sharpe = df_max_sharpe[df_max_sharpe['Peso (%)'] > 0.01].sort_values('Peso (%)', ascending=False)
                st.dataframe(df_max_sharpe.style.format({'Peso (%)': '{:.2f}'}), use_container_width=True, hide_index=True)
                
                # MÉTRICAS ORIGINAIS
                st.metric("Retorno Mensal", f"{ret_max_sharpe * 100:.4f}%")
                st.metric("Volatilidade", f"{vol_max_sharpe * 100:.4f}%")
                st.metric("Sharpe", f"{sharpe_max_sharpe:.4f}")
                
                # NOVAS MÉTRICAS - AMBOS RETORNOS ACUMULADOS
                st.metric("Retorno Acumulado Real", f"{ret_acum_real_max_sharpe * 100:.4f}%")
                st.metric("Retorno Acumulado Esperado", f"{ret_acum_esperado_max_sharpe * 100:.4f}%")

            
            # -------------------------------------------------------------
            # 🛡️ Mínima Volatilidade
            # -------------------------------------------------------------
            with col2:
                st.markdown("### 🛡️ Mínima Volatilidade")
                df_min_vol = pd.DataFrame({
                    'Ativo': [a.replace('.SA', '') for a in ativos_validos],
                    'Peso (%)': pesos_min_vol * 100
                })
                df_min_vol = df_min_vol[df_min_vol['Peso (%)'] > 0.01].sort_values('Peso (%)', ascending=False)
                st.dataframe(df_min_vol.style.format({'Peso (%)': '{:.2f}'}), use_container_width=True, hide_index=True)
                
                # MÉTRICAS ORIGINAIS
                st.metric("Retorno Mensal", f"{ret_min_vol * 100:.4f}%")
                st.metric("Volatilidade", f"{vol_min_vol * 100:.4f}%")
                st.metric("Sharpe", f"{sharpe_min_vol:.4f}")
                
                # NOVAS MÉTRICAS - AMBOS RETORNOS ACUMULADOS
                st.metric("Retorno Acumulado Real", f"{ret_acum_real_min_vol * 100:.4f}%")
                st.metric("Retorno Acumulado Esperado", f"{ret_acum_esperado_min_vol * 100:.4f}%")

            
            
            # =============================================================
            # 3) TARGET
            # =============================================================
            st.markdown("---")
            st.markdown("### 🎯 Target")
            col3_1, col3_2 = st.columns([2, 1])

            with col3_1:
                df_target = pd.DataFrame({
                    'Ativo': [a.replace('.SA', '') for a in ativos_validos],
                    'Peso (%)': w_target * 100
                })
                df_target = df_target[df_target['Peso (%)'] > 0.01].sort_values('Peso (%)', ascending=False)
                st.dataframe(df_target.style.format({'Peso (%)': '{:.2f}'}), use_container_width=True, hide_index=True)

            with col3_2:
                # MÉTRICAS ORIGINAIS
                st.metric("Retorno Mensal", f"{ret_target * 100:.4f}%")
                st.metric("Volatilidade", f"{vol_target * 100:.4f}%")
                st.metric("Sharpe", f"{sharpe_target:.4f}")
                
                # NOVAS MÉTRICAS - AMBOS RETORNOS ACUMULADOS
                st.metric("Retorno Acumulado Real", f"{ret_acum_real_target * 100:.4f}%")
                st.metric("Retorno Acumulado Esperado", f"{ret_acum_esperado_target * 100:.4f}%")

            
            
            # =============================================================
            # 4) CARTEIRAS LIMITADAS (SÓ SE limite_peso > 0)
            # =============================================================
            if limite_peso > 0:
                st.markdown("---")
                st.markdown("## 🔒 Otimizações com Limite de Peso")
                
                col_lim1, col_lim2 = st.columns(2)
                
                # ---- SHARPE LIMITADO
                with col_lim1:
                    st.markdown("### 🔒 Máximo Sharpe (Limitado)")
                    df_lim_sharpe = pd.DataFrame({
                        'Ativo': [a.replace('.SA', '') for a in ativos_validos],
                        'Peso (%)': pesos_max_sharpe_lim * 100
                    })
                    df_lim_sharpe = df_lim_sharpe[df_lim_sharpe['Peso (%)'] > 0.01].sort_values('Peso (%)', ascending=False)
                    st.dataframe(df_lim_sharpe.style.format({'Peso (%)': '{:.2f}'}), use_container_width=True, hide_index=True)
                    
                    # MÉTRICAS ORIGINAIS
                    st.metric("Retorno Mensal", f"{ret_max_sharpe_lim * 100:.4f}%")
                    st.metric("Volatilidade", f"{vol_max_sharpe_lim * 100:.4f}%")
                    st.metric("Sharpe", f"{sharpe_max_sharpe_lim:.4f}")
                    
                    # NOVAS MÉTRICAS - AMBOS RETORNOS ACUMULADOS
                    st.metric("Retorno Acumulado Real", f"{ret_acum_real_max_sharpe_lim * 100:.4f}%")
                    st.metric("Retorno Acumulado Esperado", f"{ret_acum_esperado_max_sharpe_lim * 100:.4f}%")

                
                # ---- MIN VOL LIMITADA
                with col_lim2:
                    st.markdown("### 🔒 Mínima Volatilidade (Limitada)")
                    df_lim_minvol = pd.DataFrame({
                        'Ativo': [a.replace('.SA', '') for a in ativos_validos],
                        'Peso (%)': pesos_min_vol_lim * 100
                    })
                    df_lim_minvol = df_lim_minvol[df_lim_minvol['Peso (%)'] > 0.01].sort_values('Peso (%)', ascending=False)
                    st.dataframe(df_lim_minvol.style.format({'Peso (%)': '{:.2f}'}), use_container_width=True, hide_index=True)
                    
                    # MÉTRICAS ORIGINAIS
                    st.metric("Retorno Mensal", f"{ret_min_vol_lim * 100:.4f}%")
                    st.metric("Volatilidade", f"{vol_min_vol_lim * 100:.4f}%")
                    st.metric("Sharpe", f"{sharpe_min_vol_lim:.4f}")
                    
                    # NOVAS MÉTRICAS - AMBOS RETORNOS ACUMULADOS
                    st.metric("Retorno Acumulado Real", f"{ret_acum_real_min_vol_lim * 100:.4f}%")
                    st.metric("Retorno Acumulado Esperado", f"{ret_acum_esperado_min_vol_lim * 100:.4f}%")

            
            
            # =============================================================
            # 5) MINHA CARTEIRA
            # =============================================================
            if pesos_user is not None:
                st.markdown("---")
                st.markdown("### 🌟 Minha Carteira Personalizada")
                
                col_user1, col_user2 = st.columns([2, 1])
                
                with col_user1:
                    df_user = pd.DataFrame({
                        'Ativo': [a.replace('.SA', '') for a in ativos_validos],
                        'Peso (%)': pesos_user * 100
                    })
                    df_user = df_user[df_user['Peso (%)'] > 0.01].sort_values('Peso (%)', ascending=False)
                    st.dataframe(df_user.style.format({'Peso (%)': '{:.2f}'}), use_container_width=True, hide_index=True)
                
                with col_user2:
                    # MÉTRICAS ORIGINAIS
                    st.metric("Retorno Mensal", f"{ret_user * 100:.4f}%")
                    st.metric("Volatilidade", f"{vol_user * 100:.4f}%")
                    st.metric("Sharpe", f"{sharpe_user:.4f}")
                    
                    # NOVAS MÉTRICAS - AMBOS RETORNOS ACUMULADOS
                    st.metric("Retorno Acumulado Real", f"{ret_acum_real_user * 100:.4f}%")
                    st.metric("Retorno Acumulado Esperado", f"{ret_acum_esperado_user * 100:.4f}%")

        
        with tab3:
            st.subheader("Métricas Individuais por Ativo")
            
            metricas_individuais = []
            for ativo in ativos_validos:
                r = media_retorno[ativo]
                v = np.sqrt(matriz_covariancia.loc[ativo, ativo])
                s = r / v if v != 0 else np.nan
                metricas_individuais.append({
                    'Ativo': ativo.replace('.SA', ''),
                    'Retorno Médio (%)': r * 100,
                    'Volatilidade (%)': v * 100,
                    'Sharpe': s
                })
            
            df_metricas = pd.DataFrame(metricas_individuais)
            df_metricas = df_metricas.sort_values('Sharpe', ascending=False)
            
            st.dataframe(
                df_metricas.style.format({
                    'Retorno Médio (%)': '{:.4f}',
                    'Volatilidade (%)': '{:.4f}',
                    'Sharpe': '{:.4f}'
                }).background_gradient(subset=['Sharpe'], cmap='RdYlGn'),
                use_container_width=True,
                hide_index=True
            )
            
            st.markdown("---")
            st.subheader("Matriz de Correlação")
            
            fig_corr = px.imshow(
                matriz_corr,
                labels=dict(x="Ativo", y="Ativo", color="Correlação"),
                x=[a.replace('.SA', '') for a in ativos_validos],
                y=[a.replace('.SA', '') for a in ativos_validos],
                color_continuous_scale='RdBu_r',
                aspect="auto",
                zmin=-1, zmax=1
            )
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with tab4:
            st.subheader("Informações Detalhadas")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Período de Análise", f"{data_inicio} a {data_fim}.")
            with col2:
                st.metric("Número de Ativos", len(ativos_validos))
            
            st.markdown("---")
            
            col4, col5, col6 = st.columns(3)
            with col4:
                st.metric("Tipo de Cotação", "Ajustada" if usar_ajustada else "Simples")
            with col5:
                st.metric("Tipo de Retorno", "Logarítmico" if usar_log else "Simples")
            with col6:
                st.metric("Retorno Alvo", f"{target_return * 100:.2f}%")
            
            # =============================================================
            # NOVA SEÇÃO: CONFIGURAÇÕES DE LIMITE DE PESO
            # =============================================================
            st.markdown("---")
            st.subheader("Configurações de Limite de Peso")

            col_lim1, col_lim2 = st.columns(2)

            with col_lim1:
                # Mostra o limite que o usuário escolheu (convertendo de volta para porcentagem)
                limite_escolhido_pct = limite_peso * 100
                st.metric("Limite Escolhido pelo Usuário", f"{limite_escolhido_pct:.1f}%")

            with col_lim2:
                if limite_peso > 0:
                    # Calcula o limite efetivamente aplicado
                    limite_efetivo = limite_peso * 100
                    limite_minimo_necessario = (1 / len(ativos_validos)) * 100
                    
                    if limite_efetivo < limite_minimo_necessario:
                        st.metric("Limite Efetivamente Aplicado", f"{limite_minimo_necessario:.1f}%")
                        st.caption(f"*Ajustado do limite escolhido ({limite_efetivo:.1f}%) para o mínimo necessário*")
                    else:
                        st.metric("Limite Efetivamente Aplicado", f"{limite_efetivo:.1f}%")
                        st.caption("*Limite aplicado conforme escolhido pelo usuário*")
                else:
                    st.metric("Limite Efetivamente Aplicado", "Desativado")
                    st.caption("*Carteiras limitadas não foram calculadas*")
            
            # Informação adicional sobre o limite
            if limite_peso > 0:
                st.info(f"""
                **📊 Informações sobre o limite de peso:**
                - **Número de ativos:** {len(ativos_validos)}
                - **Limite mínimo necessário:** {(1/len(ativos_validos))*100:.1f}%
                - **Status:** Carteiras limitadas **ativas**
                - **Estratégias calculadas:** Máximo Sharpe Limitado e Mínima Volatilidade Limitada
                """)
            else:
                st.info("""
                **📊 Informações sobre o limite de peso:**
                - **Status:** Carteiras limitadas **desativadas**
                - **Estratégias calculadas:** Apenas as estratégias principais (sem limite)
                """)
            
            st.markdown("---")
            st.subheader("Ativos Analisados")
            ativos_display = [a.replace('.SA', '') for a in ativos_validos]
            st.write(", ".join(ativos_display))
    
    except Exception as e:
        st.error(f"Erro durante o processamento: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

else:
    st.info("👈 Configure os parâmetros na barra lateral e clique em 'Calcular Portfólios'")
    
    st.markdown("""
    ### Sobre esta Aplicação
    
    Esta ferramenta realiza otimização de portfólio de ações utilizando a **Teoria Moderna de Portfólios** de Markowitz.
    
    #### Estratégias Implementadas:
    
    - **Máximo Sharpe**: Maximiza o retorno ajustado ao risco (melhor relação retorno/volatilidade)
    - **Mínima Volatilidade**: Minimiza o risco (desvio padrão) total do portfólio
    - **Target**: Minimiza o risco para atingir um retorno alvo específico
    - **Limite de Peso Máximo**: Opção para limitar o peso máximo de cada ativo no portfólio otimizado
    - **Carteira Personalizada**: Analise sua própria carteira comparando com as estratégias ótimas
    
    #### 📊 Entenda os Retornos Acumulados:

    Na aba **Estratégias**, você verá **DOIS** tipos de retorno acumulado:

    **🎯 Retorno Acumulado REAL**
    - **O que é**: Retorno que realmente aconteceu no período histórico
    - **Significado**: "Se eu tivesse investido nessa estratégia, quanto teria hoje?"

    **📈 Retorno Acumulado ESPERADO**  
    - **O que é**: Projeção baseada no retorno médio mensal
    - **Significado**: "Se esse retorno médio se repetir, quanto terei no futuro?"

    **💡 Por que mostrar ambos?**
    - O **Real** mostra o desempenho histórico verdadeiro
    - O **Esperado** ajuda a projetar resultados futuros
    - A diferença entre eles revela o impacto da **volatilidade** - retornos variáveis reduzem o ganho real!                

    #### Como Usar:
    
    1. Configure o período de análise (datas inicial e final, **ANO/MÊS/DIA**)
    2. Insira os tickers das ações/ETFs brasileiros (ex: PETR4, VALE3, BOVA11)
    3. Escolha entre cotação ajustada ou simples
    4. Selecione retornos logarítmicos ou simples
    5. (Opcional) Defina um limite máximo de peso por ativo para uma otimização mais diversificada
    6. Defina o retorno alvo mensal desejado
    7. (Opcional) Insira sua carteira personalizada para análise (*apenas ativos que foram selecionados para otimização anterior*)
    8. Clique em "Calcular Portfólios"
    9. Explore os resultados nas quatro abas:
       - **Fronteira Eficiente**: Visualização gráfica de todas as estratégias
       - **Estratégias**: Comparação detalhada e pesos de cada estratégia
       - **Ativos Individuais**: Métricas e correlações dos ativos
       - **Detalhes**: Informações técnicas da análise
    """)
