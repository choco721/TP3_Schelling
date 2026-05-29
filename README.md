# TP3_Schelling# Modelo de Segregación de Schelling — TP3 SyPE 2026

Simulación basada en agentes del modelo de Schelling implementada en Python.

## Descripción

Dos dinámicas sobre una grilla 50×50 con vecindad de Moore-8:

- **Dinámica 1 (preferencia por minoría):** los agentes quieren estar rodeados
  del grupo opuesto. Resulta en un sistema frustrado sin equilibrio estable.

- **Dinámica 2 (preferencia por mayoría):** los agentes quieren vecinos similares.
  Converge rápidamente a barrios segregados pese a preferencias individuales leves.

También incluye el modelo de Ising (algoritmo Metropolis) y análisis de
aplicabilidad de MCMC a segregación espacial.

## Ejecución

```bash
pip install -r requirements.txt
python src/punto1_minority.py
python src/punto2_majority.py
```

## Estructura
src/          módulos de simulación
notebooks/    desarrollo completo del TP con figuras
outputs/      gráficos generados

## Referencias

- Schelling, T.C. (1971). Dynamic models of segregation.
- Ciaburro, G. (2022). Hands-On Simulation Modeling with Python, 2nd ed. Packt.
- Downey, A.B. ModSimPy. https://allendowney.github.io/ModSimPy/

## Autores

Ancarani · Formenti · Mendes · Morenico · [tu apellido]  
Lic. Ciencias de Datos — UCA Rosario · SyPE 2026
