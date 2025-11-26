import yfinance as yf  # baixar dados do Yahoo Finance
import pandas as pd  # manipulação de tabelas
import numpy as np  # cálculos numéricos
import datetime as dt  # manipulação de datas
from scipy.optimize import minimize  # otimizador para fronteira eficiente
import matplotlib.pyplot as plt  # plotagem
import matplotlib.ticker as mtick  # formatação de eixos

# -----------------------------
# CONFIGURAÇÕES
# -----------------------------
inicio = dt.datetime(2020, 12, 1)  # data inicial
fim = dt.datetime(2025, 11, 1)  # data final

# ================================
# ENTRADA MANUAL DE ATIVOS
# ================================
def perguntar_ativos():
    print("\nDigite os ativos que deseja usar (um por linha).")
    print("Exemplo: PETR4  → não precisa colocar .SA")
    print("Quando terminar, digite FIM\n")

    ativos = []
    while True:
        linha = input().strip().upper()
        if linha == "FIM":
            break
        if linha == "":
            continue
        ativos.append(linha)

    ativos = list(dict.fromkeys(ativos))  # remove duplicatas
    return ativos


def validar_ativos(lista):
    ativos_ok = []
    ativos_errados = []

    for ticker in lista:
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

    return ativos_ok, ativos_errados


# ============================
# LOOP PRINCIPAL AJUSTADO
# ============================

ativos_digitados = perguntar_ativos()

while True:
    ativos_validos, ativos_invalidos = validar_ativos(ativos_digitados)

    print("\n-------------------------------")
    print("ATIVOS ATUAIS :", ativos_digitados)
    print("ATIVOS VÁLIDOS:", ativos_validos)
    print("ATIVOS INVÁLIDOS:", ativos_invalidos)
    print("-------------------------------\n")

    # CASO NÃO HAJA ERROS
    if not ativos_invalidos:
        print("✔ Todos os ativos são válidos! Prosseguindo...\n")
        break

    # CASO HAJA ATIVOS INVÁLIDOS
    print("⚠ Alguns ativos não foram encontrados no Yahoo Finance:\n")
    for errado in ativos_invalidos:
        print(f"   → {errado} (não localizado)")

    print("\nVocê deseja CORRIGIR somente os ativos inválidos? (s/n)")
    resp = input().strip().lower()

    # CORRIGIR SOMENTE OS ERRADOS
    if resp == "s":
        print("\nDigite as correções para os seguintes tickers:")

        for ticker_errado in ativos_invalidos:
            novo = input(f"{ticker_errado} → ").strip().upper()

            if novo == "":
                print("   (mantido como inválido)")
                continue

            # remover o inválido e adicionar o novo
            ativos_digitados.remove(ticker_errado)
            ativos_digitados.append(novo)

        # volta ao início do while para validar de novo
        continue

    # NÃO QUER CORRIGIR → remove inválidos e segue com os válidos
    else:
        print("\nProsseguindo APENAS com os ativos válidos.\n")
        # remove os inválidos definitivamente
        for errado in ativos_invalidos:
            ativos_digitados.remove(errado)

        # validar pela última vez
        ativos_validos, _ = validar_ativos(ativos_digitados)
        break


# LISTA FINAL USADA PELO RESTANTE DO CÓDIGO
lista_acoes = ativos_validos


# Escolha se quer preços ajustados (considera proventos) ou não
usar_ajustada = True  # True -> usa 'Adj Close' (com dividendos/splits)
if usar_ajustada == True:
    print("✅Cotação Ajustada a Proventos, Bonificações e Splits✅")
else:
    print("❌Cotação Simples❌")

# Número de carteiras simuladas (Monte Carlo)
numero_carteiras = 100000

# -----------------------------
# BAIXAR DADOS (yfinance)
# -----------------------------
# aqui passamos auto_adjust conforme usar_ajustada para evitar o warning e respeitar a escolha
dados = yf.download(lista_acoes, start=inicio, end=fim, auto_adjust=usar_ajustada, progress=False)

# Seleciona preços (ajustado ou não) e trata MultiIndex de colunas
# yfinance retorna MultiIndex com colunas (Open, High, Low, Close, Adj Close, Volume) em algumas versões
if usar_ajustada:
    # preferimos 'Adj Close' quando disponível
    if hasattr(dados.columns, "levels") and "Adj Close" in dados.columns.levels[0]:
        precos = dados["Adj Close"].copy()
    else:
        precos = dados["Close"].copy()
else:
    precos = dados["Close"].copy()

# Se houver MultiIndex, pegamos somente o nível com os tickers (garante colunas planas)
if isinstance(precos.columns, pd.MultiIndex):
    precos.columns = precos.columns.get_level_values(0)

# -----------------------------
# REAMOSTRAGEM MENSAL (ME = month-end)
# -----------------------------
# usar 'ME' (month end) — mantive igual ao seu código
precos_mensais = precos.resample("ME").last()

# -----------------------------
# FILTRAR ATIVOS COM PELO MENOS 24 MESES DE DADOS
# -----------------------------
meses_por_ativo = precos_mensais.count()
ativos_validos_24m = meses_por_ativo[meses_por_ativo >= 24].index
precos_mensais = precos_mensais[ativos_validos_24m]

if len(ativos_validos_24m) == 0:
    raise SystemExit("Nenhum ativo possui ao menos 24 meses de histórico. Ajuste a lista ou a janela temporal.")

print("Ativos usados após filtro de 24 meses:", list(ativos_validos_24m))

# -----------------------------
# ESCOLHA DO TIPO DE RETORNO
# -----------------------------
print("\nDeseja usar retornos logarítmicos? (s = log, n = simples)")
resp_retorno = input().strip().lower()

usar_log = (resp_retorno == "s")

# -----------------------------
# CÁLCULO DOS RETORNOS MENSAIS
# -----------------------------
if usar_log:
    print("→ Usando retornos logarítmicos.")
    retornos = np.log(precos_mensais / precos_mensais.shift(1))
else:
    print("→ Usando retornos simples (percentuais).")
    retornos = precos_mensais.pct_change()

# limpar NaN
retornos = retornos.dropna(how="all")
retornos = retornos.dropna(axis=1, how="all")

ativos_validos = list(retornos.columns)
n = len(ativos_validos)

if n == 0:
    raise SystemExit("Nenhum ativo com retornos válidos após cálculo.")


# -----------------------------
# MÉTRICAS: média e covariância
# -----------------------------
media_retorno = retornos.mean()

matriz_covariancia = retornos.cov()

matriz_corr = retornos.corr()
matriz_corr_vals = matriz_corr.values

vol_individual = np.sqrt(np.diag(matriz_covariancia.values))


# =============================================================================
# MÉTRICAS INDIVIDUAIS DE CADA ATIVO
# =============================================================================
print("\n" + "="*80)
print(" INFORMAÇÕES INDIVIDUAIS POR ATIVO")
print("="*80)
print(f"{'Ativo':<10} {'Retorno Médio':>15} {'Volatilidade':>15} {'Sharpe':>10}")
print("-"*80)

for ativo in ativos_validos:
    r = media_retorno[ativo]
    v = np.sqrt(matriz_covariancia.loc[ativo, ativo])
    s = r / v if v != 0 else np.nan
    print(f"{ativo:<10} {r*100:>14.4f}% {v*100:>14.4f}% {s:>10.4f}")


# -----------------------------
# FUNÇÕES BASE DE OTIMIZAÇÃO
# -----------------------------
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

bounds = tuple((0.0, 1.0) for _ in range(len(ativos_validos)))
w0 = np.ones(len(ativos_validos)) / len(ativos_validos)

# -----------------------------
# MONTE CARLO (carteiras aleatórias)
# -----------------------------
tabela_retornos = np.zeros(numero_carteiras)
T_tabela_pesos = np.zeros((numero_carteiras, n))
tabela_volatilidades = np.zeros(numero_carteiras)
tabela_sharpe = np.zeros(numero_carteiras)

# Monte Carlo loop
for k in range(numero_carteiras):
    w = np.random.random(n)
    w /= w.sum()
    T_tabela_pesos[k, :] = w
    tabela_retornos[k] = np.dot(w, media_retorno[ativos_validos])
    tabela_volatilidades[k] = np.sqrt(np.dot(w.T, np.dot(matriz_covariancia.loc[ativos_validos, ativos_validos].values, w)))
    tabela_sharpe[k] = tabela_retornos[k] / tabela_volatilidades[k] if tabela_volatilidades[k] != 0 else -np.inf

# carteira (simulada) de máximo Sharpe (apenas da simulação)
indice_max_sharpe = np.nanargmax(tabela_sharpe)
pesos_simul_max_sharpe = T_tabela_pesos[indice_max_sharpe]
ret_sim_max_sharpe = tabela_retornos[indice_max_sharpe]
vol_sim_max_sharpe = tabela_volatilidades[indice_max_sharpe]
sharpe_sim_max_sharpe = tabela_sharpe[indice_max_sharpe]

# -----------------------------
# OTIMIZAÇÕES EXATAS (SLSQP)
# -----------------------------
# Max Sharpe (otimização direta)
res_maxsharpe = minimize(
    sharpe_negativo,
    w0,
    method="SLSQP",
    bounds=bounds,
    constraints={'type': 'eq', 'fun': soma_pesos}
)
if not res_maxsharpe.success:
    pesos_max_sharpe = w0.copy()
else:
    pesos_max_sharpe = res_maxsharpe.x
ret_max_sharpe = retorno_portfolio(pesos_max_sharpe)
vol_max_sharpe = risco_portfolio(pesos_max_sharpe)
sharpe_max_sharpe = ret_max_sharpe / vol_max_sharpe if vol_max_sharpe != 0 else np.nan

# Min Variância (otimização direta)
res_minvar = minimize(
    risco_portfolio,
    w0,
    method="SLSQP",
    bounds=bounds,
    constraints={'type': 'eq', 'fun': soma_pesos}
)
if not res_minvar.success:
    pesos_min_vol = w0.copy()
else:
    pesos_min_vol = res_minvar.x
ret_min_vol = retorno_portfolio(pesos_min_vol)
vol_min_vol = risco_portfolio(pesos_min_vol)
sharpe_min_vol = ret_min_vol / vol_min_vol if vol_min_vol != 0 else np.nan

# -----------------------------
# EXTREMOS DE RETORNO (ativos únicos)
# -----------------------------
retornos_individuais = media_retorno[ativos_validos].values
idx_max_ret = np.argmax(retornos_individuais)
idx_min_ret = np.argmin(retornos_individuais)

pesos_max_retorno = np.zeros(len(ativos_validos))
pesos_min_retorno = np.zeros(len(ativos_validos))
pesos_max_retorno[idx_max_ret] = 1.0
pesos_min_retorno[idx_min_ret] = 1.0

ret_max_retorno = retornos_individuais[idx_max_ret]
vol_max_retorno = risco_portfolio(pesos_max_retorno)
sharpe_max_retorno = ret_max_retorno / vol_max_retorno if vol_max_retorno != 0 else np.nan

ret_min_retorno = retornos_individuais[idx_min_ret]
vol_min_retorno = risco_portfolio(pesos_min_retorno)
sharpe_min_retorno = ret_min_retorno / vol_min_retorno if vol_min_retorno != 0 else np.nan

# -----------------------------
# DR1 - Diversification Ratio (otimização)
# -----------------------------
def objetivo_dr1(w):
    w = np.array(w)
    numerador = np.dot(w, vol_individual)
    denom = risco_portfolio(w)
    if denom == 0:
        return 1e6
    DR = numerador / denom
    return -DR

res_dr1 = minimize(objetivo_dr1, w0, method="SLSQP", bounds=bounds, constraints={'type': 'eq', 'fun': soma_pesos})
if not res_dr1.success:
    pesos_dr1 = w0.copy()
else:
    pesos_dr1 = res_dr1.x
ret_dr1 = retorno_portfolio(pesos_dr1)
vol_dr1 = risco_portfolio(pesos_dr1)
sharpe_dr1 = ret_dr1 / vol_dr1 if vol_dr1 != 0 else np.nan

# -----------------------------
# CR1 - carteira de mínima correlação média (otimização)
# -----------------------------
def correlacao_media(w):
    w = np.array(w)
    # usa matriz de correlação completa (valores entre -1 e 1)
    return float(np.sum(np.outer(w, w) * matriz_corr_vals))

res_cr1 = minimize(correlacao_media, w0, method="SLSQP", bounds=bounds, constraints={'type': 'eq', 'fun': soma_pesos})
if not res_cr1.success:
    pesos_cr1 = w0.copy()
else:
    pesos_cr1 = res_cr1.x
ret_cr1 = retorno_portfolio(pesos_cr1)
vol_cr1 = risco_portfolio(pesos_cr1)
corr_cr1 = correlacao_media(pesos_cr1)
sharpe_cr1 = ret_cr1 / vol_cr1 if vol_cr1 != 0 else np.nan

# -----------------------------
# CR1-RET - maximizar Retorno / Correlação (otimização)
# -----------------------------
def objetivo_cr1_ret(w):
    w = np.array(w)
    r = retorno_portfolio(w)
    c = correlacao_media(w)
    if c <= 0:
        return 1e6
    return -(r / c)

res_cr1_ret = minimize(objetivo_cr1_ret, w0, method="SLSQP", bounds=bounds, constraints={'type': 'eq', 'fun': soma_pesos})
if not res_cr1_ret.success:
    pesos_cr1_ret = w0.copy()
else:
    pesos_cr1_ret = res_cr1_ret.x
ret_cr1_ret = retorno_portfolio(pesos_cr1_ret)
vol_cr1_ret = risco_portfolio(pesos_cr1_ret)
corr_cr1_ret = correlacao_media(pesos_cr1_ret)
ratio_cr1_ret = ret_cr1_ret / corr_cr1_ret if corr_cr1_ret != 0 else np.nan

# -----------------------------
# TARGET X% ao mês - minimizar volatilidade para retorno alvo (BLOCO ROBUSTO)
# -----------------------------

# --- DEFINIR ret_min e ret_max CORRETAMENTE ---
# Calcula os limites de retorno possíveis
if 'tabela_retornos' in globals():
    ret_min = float(np.nanmin(tabela_retornos))
    ret_max = float(np.nanmax(tabela_retornos))
else:
    # fallback: usa mínimos/máximos dos retornos individuais
    ret_min = float(media_retorno[ativos_validos].min())
    ret_max = float(media_retorno[ativos_validos].max())

# mostra os limites ao usuário
print(f"\nLimites práticos de retorno (mensal) detectados — mínimo: {ret_min*100:.4f}%, máximo: {ret_max*100:.4f}%")

# pede o retorno alvo ao usuário (entrada segura)
print("\nDigite o retorno alvo mensal desejado (ex: 0.008 para 0,8%). Deixe em branco para usar 0.01 (1%):")
entrada = input().strip()
if entrada == "":
    target_return = 0.01
else:
    try:
        target_return = float(entrada.replace(",", "."))
    except Exception:
        print("Entrada inválida. Usando retorno alvo default de 1% ao mês.")
        target_return = 0.01

# garante que o target esteja dentro do intervalo praticável
if target_return < ret_min:
    print(f"\n⚠️ Retorno alvo ({target_return*100:.4f}%) abaixo do mínimo observável ({ret_min*100:.4f}%). Ajustando para {ret_min*100:.4f}%.")
    target_return = ret_min
elif target_return > ret_max:
    print(f"\n⚠️ Retorno alvo ({target_return*100:.4f}%) acima do máximo observável ({ret_max*100:.4f}%). Ajustando para {ret_max*100:.4f}%.")
    target_return = ret_max

# restrição de retorno-alvo (usa média histórica)
def constraint_retorno(w):
    return float(np.dot(w, media_retorno[ativos_validos]) - target_return)

# otimização: minimizar risco sujeita a soma dos pesos = 1 e retorno = target_return
res_target = minimize(
    risco_portfolio,
    w0,
    method="SLSQP",
    bounds=bounds,
    constraints=[
        {'type': 'eq', 'fun': soma_pesos},
        {'type': 'eq', 'fun': constraint_retorno}
    ],
    options={'ftol': 1e-12, 'maxiter': 1000}
)

if not res_target.success:
    print("\n⚠️ Otimização Target falhou ou não convergiu. Usando pesos uniformes como fallback.")
    w_target = w0.copy()
else:
    w_target = res_target.x

# métricas resultantes
ret_target = retorno_portfolio(w_target)
vol_target = risco_portfolio(w_target)
sharpe_target = ret_target / vol_target if vol_target != 0 else np.nan

print("\n===== RESULTADO TARGET =====")
for ativo, peso in zip(ativos_validos, w_target):
    print(f"{ativo}: {peso*100:.2f}%")
print(f"Retorno mensal (alvo): {ret_target*100:.4f}%")
print(f"Volatilidade mensal (alvo): {vol_target*100:.4f}%")
print(f"Sharpe mensal (alvo): {sharpe_target:.4f}")



# -----------------------------
# Fronteira eficiente (otimização para vários retornos-alvo)
# -----------------------------
ret_min = tabela_retornos.min()
ret_max = tabela_retornos.max()
alvo_rets = np.linspace(ret_min, ret_max, 80)
fronteira_vols = []
fronteira_rets = []
fronteira_pesos = []

for r in alvo_rets:
    cons_r = (
        {'type': 'eq', 'fun': soma_pesos},
        {'type': 'eq', 'fun': lambda w, r=r: np.dot(w, media_retorno[ativos_validos]) - r}
    )
    res = minimize(risco_portfolio, w0, method='SLSQP', bounds=bounds, constraints=cons_r)
    if res.success:
        fronteira_pesos.append(res.x)
        fronteira_vols.append(risco_portfolio(res.x))
        fronteira_rets.append(np.dot(res.x, media_retorno[ativos_validos]))

# -----------------------------
# ENTRADA: carteira manual do usuário (opcional)
# -----------------------------
print("\nDeseja analisar sua própria carteira? (s/n)")
resp = input().strip().lower()

pesos_user = None
ret_user = vol_user = sharpe_user = corr_user = None

if resp == "s":
    print("\nDigite sua carteira no formato: TICKER PESO (em %) — um por linha. Ex: PETR4 17")
    print("Quando terminar, digite FIM\n")
    carteira_raw = {}
    while True:
        linha = input().strip()
        if linha.lower() == "fim":
            break
        if len(linha.split()) != 2:
            print("Formato inválido. Use: TICKER PESO")
            continue
        ticker, peso_str = linha.split()
        try:
            peso = float(peso_str.replace(",", "."))
        except:
            print("Peso inválido.")
            continue
        ticker_full = ticker.upper()
        if not ticker_full.endswith(".SA"):
            ticker_full += ".SA"
        carteira_raw[ticker_full] = peso

    pesos_user = np.zeros(len(ativos_validos))
    for i, ativo in enumerate(ativos_validos):
        if ativo in carteira_raw:
            pesos_user[i] = carteira_raw[ativo]
    soma = pesos_user.sum()
    if soma == 0:
        print("\nNenhum ativo informado corresponde aos ativos válidos do código.")
        pesos_user = None
    else:
        pesos_user = pesos_user / soma
        ret_user = retorno_portfolio(pesos_user)
        vol_user = risco_portfolio(pesos_user)
        sharpe_user = ret_user / vol_user if vol_user != 0 else np.nan
        corr_user = correlacao_media(pesos_user)

        print("\n===== 📌 MINHA CARTEIRA =====")
        for ativo, peso in zip(ativos_validos, pesos_user):
            print(f"{ativo}: {peso*100:.2f}%")
        print(f"\nRetorno esperado (média log): {ret_user*100:.4f}% ao mês")
        print(f"Volatilidade mensal: {vol_user*100:.4f}%")
        print(f"Sharpe mensal: {sharpe_user:.4f}")
        print(f"Correlação média ponderada: {corr_user:.6f}")

# -----------------------------
# IMPRIMIR CARTEIRAS (pesos em %)
# -----------------------------
def print_portfolio(title, pesos, ret, vol, sharpe):
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)
    for ativo, peso in zip(ativos_validos, pesos):
        print(f"{ativo:10s}: {peso*100:7.2f}%")
    print(f"\n Retorno mensal: {ret*100:7.4f}%")
    print(f" Volatilidade mensal: {vol*100:7.4f}%")
    print(f" Sharpe (mês): {sharpe:7.4f}")

# print_portfolio("MAIOR RETORNO (ativo único)", pesos_max_retorno, ret_max_retorno, vol_max_retorno, sharpe_max_retorno)
# print_portfolio("MENOR RETORNO (ativo único)", pesos_min_retorno, ret_min_retorno, vol_min_retorno, sharpe_min_retorno)
print_portfolio("MÁXIMO SHARPE (otimização)", pesos_max_sharpe, ret_max_sharpe, vol_max_sharpe, sharpe_max_sharpe)
print_portfolio("MÍNIMA VOLATILIDADE (opt)", pesos_min_vol, ret_min_vol, vol_min_vol, sharpe_min_vol)
# print_portfolio("CR1 - MÍNIMA CORRELAÇÃO (opt)", pesos_cr1, ret_cr1, vol_cr1, sharpe_cr1)
# print(f" Correlação média (CR1): {corr_cr1:.6f}")
# print_portfolio("CR1-RET - MAX RET / CORR (opt)", pesos_cr1_ret, ret_cr1_ret, vol_cr1_ret, ret_cr1_ret/vol_cr1_ret if vol_cr1_ret!=0 else np.nan)
# print(f" Correlação média (CR1-RET): {corr_cr1_ret:.6f} | Ret/Corr: {ratio_cr1_ret:.6f}")
# print_portfolio("DR1 - MÁXIMA DIVERSIFICAÇÃO (opt)", pesos_dr1, ret_dr1, vol_dr1, sharpe_dr1)
print_portfolio("TARGET (min vol p/ retorno alvo)", w_target, ret_target, vol_target, sharpe_target)

# =====================================================
# 🔥 GRÁFICO COMPLETO CORRIGIDO
# =====================================================
fig, ax = plt.subplots(figsize=(14, 10))

# Monte Carlo
sc = ax.scatter(
    tabela_volatilidades*100,
    tabela_retornos*100,
    c=tabela_sharpe,
    cmap='viridis',
    s=12,
    alpha=0.40
)

# ------ PONTOS IMPORTANTES ------
ax.scatter(vol_max_sharpe*100, ret_max_sharpe*100,
           color='red', s=150, label='Máx Sharpe (ótimo)')

ax.scatter(vol_min_vol*100, ret_min_vol*100,
           color='blue', s=150, label='Menor Risco (ótimo)')

ax.scatter(vol_max_retorno*100, ret_max_retorno*100,
           color='black', marker='X', s=180, label='Máx Retorno (ativo único)')

ax.scatter(vol_min_retorno*100, ret_min_retorno*100,
           color='gray', marker='X', s=180, label='Mín Retorno (ativo único)')

# # CR1 / CR1-RET / DR1 / TARGET
# ax.scatter(vol_cr1*100, ret_cr1*100,
#            color='cyan', marker='s', s=160, label='CR1')

# ax.scatter(vol_cr1_ret*100, ret_cr1_ret*100,
#            color='cyan', marker='P', s=170, label='CR1-RET')

# ax.scatter(vol_dr1*100, ret_dr1*100,
#            color='magenta', marker='D', s=180, label='DR1')

ax.scatter(vol_target*100, ret_target*100,
           color='purple', marker='o', s=180, label='Target')

# fronteira eficiente
if len(fronteira_vols) > 0:
    ax.plot(np.array(fronteira_vols)*100,
            np.array(fronteira_rets)*100,
            color='orange', linewidth=3, label='Fronteira Eficiente (Markowitz)')

# Carteira do usuário, se existir
if pesos_user is not None:
    ax.scatter(vol_user*100, ret_user*100,
               color='white', edgecolor='black',
               marker='*', s=260, label='Minha Carteira')

# ------ Eixos ------
ax.set_xlabel("Volatilidade Mensal (%)")
ax.set_ylabel("Retorno Mensal (%)")
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
ax.yaxis.set_major_formatter(mtick.PercentFormatter())

ax.grid(True)
ax.legend(loc='upper left')
plt.title("Carteiras + Fronteira Eficiente")

plt.show()


# --------------------------------------------------------------
# FIM DO BLOCO
# --------------------------------------------------------------

