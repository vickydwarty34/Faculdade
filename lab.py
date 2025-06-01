#ir no prompt e digitar pip install numpy scipy matplotlib  ; pip install matplotlib; pip install pandas openpyxl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.odr import Model, RealData, ODR


# === Leitura dos dados do Excel ===
arquivo_excel = r'C:\Users\Victória\Documents\dados.xlsx'  #Altere pelo local do seu arquivo
dados = pd.read_excel(arquivo_excel)

# === Extração dos dados ===                      #coloque pelo nome que está o seu arquivo do excel
R = dados['r'].values
R_err = dados['erro r'].values
l = dados['l'].values
l_err = dados['erro l'].values
A = dados['a'].values
A_err = dados['erro a'].values

# === Cálculo da variável independente: x = l / A ===
x = l / A

# Propagação da incerteza em x = l / A:
x_err = np.sqrt((l_err / A)**2 + (l * A_err / A**2)**2)

# === Modelo linear R = ρ · (l / A) ===
def modelo_linear(B, x):
    return B[0] * x  # B[0] = ρ

modelo = Model(modelo_linear)

# === Ajuste ODR (Orthogonal Distance Regression) ===
dados_odr = RealData(x, R, sx=x_err, sy=R_err)
odr = ODR(dados_odr, modelo, beta0=[1.0])
resultado = odr.run()

# === Resultados ===
rho = resultado.beta[0]
rho_err = resultado.sd_beta[0]

print(f"Resistividade elétrica ρ = ({rho:.4e} ± {rho_err:.4e}) Ω·m")

# === Geração do gráfico ===
x_plot = np.linspace(min(x), max(x), 100)
y_plot = modelo_linear([rho], x_plot)

plt.errorbar(x, R, xerr=x_err, yerr=R_err, fmt='o', capsize=3, label='Dados experimentais')
plt.plot(x_plot, y_plot, 'r-', label=f'Ajuste linear\nρ = ({rho:.2e} ± {rho_err:.2e}) Ω·m')
plt.xlabel('l / A (m⁻¹)')
plt.ylabel('R (Ω)')
plt.title('Ajuste Linear: R = ρ · (l / A)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
