# Análisis de calidad del aire de Águilas en marzo de 2024

Este proyecto reproduce la preparación, transformación, clasificación y visualización de los datos horarios de la estación Águilas (AGU). El cálculo de PM10 y PM2.5 usa el promedio móvil ponderado de 12 horas y los factores de ajuste de la NOM-172-SEMARNAT-2023. El reporte diario usa promedios de 24 horas para partículas y el máximo horario de O3.

## Archivos requeridos

- `BD_2024(2).xlsx`, con las hojas `Data` y `Param`.
- `analisis_agu_marzo_2024.py`.

## Preparación del entorno

Se recomienda Python 3.11 o 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Ejecución

Coloca el Excel en la misma carpeta del proyecto y ejecuta:

```powershell
python analisis_agu_marzo_2024.py --input "BD_2024(2).xlsx" --output "resultados"
```

El programa genera:

- datos horarios procesados;
- resumen diario y categorías;
- diagnóstico de completitud;
- tabla del periodo representativo del 27 y 28 de marzo;
- resumen de resultados en JSON;
- siete visualizaciones en PNG.

## Decisiones metodológicas principales

- Se conserva la marca de tiempo de `DATE`; la columna `HOUR` se usa para comprobar consistencia.
- Se normaliza el espacio final del encabezado `MES ` sin modificar los datos de origen.
- PM2.5 se convierte a tipo numérico y los valores ausentes permanecen como faltantes.
- No se imputan RH, WS ni WD porque falta el 100 % de sus observaciones en marzo.
- El NowCast mantiene las posiciones de las horas faltantes y exige al menos dos datos en las tres horas más recientes.
- Se aplican factores de ajuste de 0.714 para PM10 y 0.694 para PM2.5.
- Para marzo de 2024 se usan los intervalos de partículas de la columna aplicable a partir de enero de 2024.
- La categoría global es la categoría de mayor deterioro entre PM10, PM2.5 y O3. Los empates se conservan como corresponsabilidad.

## Fuente normativa

Secretaría de Medio Ambiente y Recursos Naturales. (2024, 25 de enero). *Norma Oficial Mexicana NOM-172-SEMARNAT-2023, Lineamientos para la obtención y comunicación del índice de calidad del aire y riesgos a la salud*. Diario Oficial de la Federación. https://www.dof.gob.mx/nota_detalle.php?codigo=5715154&fecha=25/01/2024

