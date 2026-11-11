import numpy as np
import matplotlib.pyplot as plt

parametros = dict(
    temp_maxima=4.0,               # Tmax
    tasa_calentamiento_oceanico=0.30,     # Roce
    enfriamiento_por_evaporacion=0.05,    # alpha
    eficiencia_evaporacion=0.20,          # beta
    consumo_humedad_por_lluvia=0.10,      # gamma
    conversion_humedad_a_lluvia=0.40,     # delta
    disipacion_lluvia=0.70,               # sigma
    amplitud_estacional=0.45,             # A
    frecuencia_anual=2 * np.pi / 12,      # omega
)

def velocidades_de_cambio(tiempo, estado, parametros):
    temp, humedad, precipitacion = estado
    pr = parametros

    tasa_crecimiento_termico = (pr["tasa_calentamiento_oceanico"]
        + pr["amplitud_estacional"] * np.cos(pr["frecuencia_anual"] * tiempo - np.pi))

    cambio_temp = (tasa_crecimiento_termico * temp
        * (1 - temp / pr["temp_maxima"])
        - pr["enfriamiento_por_evaporacion"] * humedad)

    cambio_humedad = (pr["eficiencia_evaporacion"] * temp**2
        - pr["consumo_humedad_por_lluvia"] * humedad * precipitacion)

    cambio_precipitacion = (pr["conversion_humedad_a_lluvia"] * humedad**2
        - pr["disipacion_lluvia"] * precipitacion)

    return np.array([cambio_temp, cambio_humedad, cambio_precipitacion])


def euler(estado_inicial, paso, tiempo_final, parametros):
    cantidad_pasos = round(tiempo_final / paso)

    tiempos = np.zeros(cantidad_pasos + 1)
    estados = np.zeros((cantidad_pasos + 1, 3))   
    estados[0] = estado_inicial

    for n in range(cantidad_pasos):
        velocidades = velocidades_de_cambio(tiempos[n], estados[n], parametros)
        estados[n + 1] = estados[n] + paso * velocidades
        tiempos[n + 1] = tiempos[n] + paso

    return tiempos, estados


#tiempos, estados = euler([1.0, 0.0, 0.0], paso=1.0, tiempo_final=60, parametros=parametros)

# plt.plot(tiempos, estados[:, 0], label="Temperatura")
# plt.plot(tiempos, estados[:, 1], label="Humedad")
# plt.plot(tiempos, estados[:, 2], label="Precipitación")
# plt.xlabel("Tiempo [meses]")
# plt.legend()
# plt.show()

## ----------------------
## 1A. Solución de Referencia
## ----------------------
paso_referencia = 2 ** -12

tiempos_ref, estados_ref = euler([1.0, 0.0, 0.0], paso_referencia, 60, parametros)

## grafico
# fig, ejes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)

# ejes[0].plot(tiempos_ref, estados_ref[:, 0], color="tab:red")
# ejes[0].set_ylabel("T [°C]")

# ejes[1].plot(tiempos_ref, estados_ref[:, 1], color="tab:blue")
# ejes[1].set_ylabel("H [u.h.]")

# ejes[2].plot(tiempos_ref, estados_ref[:, 2], color="tab:green")
# ejes[2].set_ylabel("P [u.p.]")
# ejes[2].set_xlabel("Tiempo [meses]")

# for eje in ejes:
#     eje.grid(alpha=0.3)
#     eje.set_xlim(0, 60)

# fig.suptitle(f"Solución de referencia (Euler, h = 2$^{{-12}}$ meses)")
# fig.tight_layout()
# fig.savefig("solucion_referencia.png", dpi=200)
# plt.show()

## ----------------------
## 1B. Análisis de Error
## ----------------------

pasos = [2**-6, 2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 1, 2]
errores = []

for paso in pasos:
    tiempos, estados = euler([1.0, 0.0, 0.0], paso, 60, parametros)

    # tomo solo los pasos q coninciden con la solución de referencia
    salto = round(paso / paso_referencia)
    temp_ref_coincidente = estados_ref[::salto, 0]

    error = np.max(np.abs(estados[:, 0] - temp_ref_coincidente))
    errores.append(error)

print(f"{'Paso':>10} | {'Error':>12}")
for paso, error in zip(pasos, errores):
    print(f"{paso:>10.5f} | {error:>12.6f}")

## ----------------------
## 1C. Orden
## ----------------------
print(f"{'Paso':>10} | {'Error':>12} | {'Orden p':>8}")
print(f"{pasos[0]:>10.5f} | {errores[0]:>12.6f} | {'-':>8}")

for i in range(1, len(pasos)):
    if pasos[i] > 1:      #2 no converge
        break
    p = np.log(errores[i] / errores[i - 1]) / np.log(2) #revisar ec
    print(f"{pasos[i]:>10.5f} | {errores[i]:>12.6f} | {p:>8.3f}")


## ----------------------
## 1D. Encontrar Inestabilidad
## ----------------------

pasos_d = [1, 1.0625, 1.125, 1.1875, 1.25, 1.3125, 1.375, 1.5, 2]

print(f"{'Paso':>8} | {'Error':>12} | {'Error/Paso':>10}")
for paso in pasos_d:
    tiempos, estados = euler([1.0, 0.0, 0.0], paso, 60, parametros)

    salto = round(paso / paso_referencia)
    temp_ref_coincidente = estados_ref[::salto, 0]

    comunes = min(len(tiempos), len(temp_ref_coincidente))
    error = np.max(np.abs(estados[:comunes, 0] - temp_ref_coincidente[:comunes]))

    print(f"{paso:>8.4f} | {error:>12.4g} | {error / paso:>10.4f}")

## ----------------------
## 2A.
## ----------------------

paso_p2 = 2**-6  

tiempos, estados = euler([1.0, 0.0, 0.0], paso_p2, 60, parametros)

ultimo_anio = tiempos >= 48
t_ult = tiempos[ultimo_anio]
temp_ult = estados[ultimo_anio, 0]
precip_ult = estados[ultimo_anio, 2]

# Severidad de la lluvia
P_max = np.max(precip_ult)
P_min = np.min(precip_ult)
P_med = np.mean(precip_ult)
severidad = (P_max - P_min) / P_med

# Desfas e
t_max_temp = t_ult[np.argmax(temp_ult)]
despues = t_ult >= t_max_temp
t_max_precip = t_ult[despues][np.argmax(precip_ult[despues])]
desfase = t_max_precip - t_max_temp

print(f"P_max = {P_max:.4f}, P_min = {P_min:.4f}, P_med = {P_med:.4f}")
print(f"Severidad S_P = {severidad:.4f}")
print(f"Máximo de T en t = {t_max_temp:.3f}, máximo de P en t = {t_max_precip:.3f}")
print(f"Desfase = {desfase:.3f} meses")

fig, ejes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
nombres = ["T [°C]", "H [u.h.]", "P [u.p.]"]
colores = ["tab:red", "tab:blue", "tab:green"]
for i in range(3):
    ejes[i].plot(tiempos, estados[:, i], color=colores[i])
    ejes[i].set_ylabel(nombres[i])
    ejes[i].axvspan(48, 60, color="gray", alpha=0.15)   # marca el último año
    ejes[i].grid(alpha=0.3)
ejes[2].set_xlabel("Tiempo [meses]")
fig.suptitle("Solución con parámetros nominales (h = 1/64)")
fig.tight_layout()
fig.savefig("parte2a_completo.png", dpi=200)

fig, ejes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
for i in range(3):
    ejes[i].plot(t_ult, estados[ultimo_anio, i], color=colores[i])
    ejes[i].set_ylabel(nombres[i])
    ejes[i].grid(alpha=0.3)
ejes[0].axvline(t_max_temp, color="tab:red", linestyle="--")
ejes[2].axvline(t_max_precip, color="tab:green", linestyle="--")
ejes[2].set_xlabel("Tiempo [meses]")
fig.suptitle("Último ciclo anual (48 ≤ t ≤ 60)")
fig.tight_layout()
fig.savefig("parte2a_ultimo_anio.png", dpi=200)
plt.show()