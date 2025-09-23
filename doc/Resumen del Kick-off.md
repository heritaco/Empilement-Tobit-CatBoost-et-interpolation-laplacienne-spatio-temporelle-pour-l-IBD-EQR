Resumen del kick-off del Datatón (UDLAP)

**Objetivo**

* Clasificar la calidad ecológica del agua.
* Libre elección de técnicas ML; justificar selección y comparación.
* Metodología requerida: **CRISP-DM** (al inicio alguien dijo “CRISPR-CM”, pero el estándar es CRISP-DM).

**Fechas y etapas**

* **Etapa 1 (trabajo y envíos):** **lun 22-sep-2025** → **jue 16-oct-2025**.
* **Selección de finalistas:** **jue 16-oct** → **sáb 18-oct-2025**.

  * Criterios: (i) **accuracy** del sitio web, (ii) **video** ≤ 8 min.
* **Etapa 2 (presentación final CRISP-DM):** finalistas presentan con jurado.
* **Premiación:** **vie 24-oct-2025, 17:30**, UDLAP, en el cierre del 2º Congreso Nacional de Ciencia de Datos.

**Datos**

* Fuente: instituto francés de tratamiento de agua (vía Univ. de Lorraine).
* Múltiples archivos; unión por identificador común (**Sampling Operation Code**).
* Variable objetivo operativa: **IBD\_EQR\_status** ∈ {“muy mala”…“muy buena”}.
* Variables relacionadas:

  * **IBD** ∈ \[0,20] → via fórmula se obtiene **IBD-EQR** → se discretiza a **status**.
  * **Importante:** **NO** usar **IBD**, **IBD-EQR** ni **IBD\_EQR\_status** como predictores.
* Notas: la conversión **IBD-EQR → status** depende de **región**; revisar README.
* Un archivo “Tax Code” está aparte; el README describe cada columna.

**Validación y envíos**

* Sitio web con **leaderboard** público.
* Formato de **CSV de predicciones**:

  * columnas: **Sampling Operation Code**, **IBD\_EQR\_status**.
* **Límite:** **2 envíos por día por equipo** con **API key** individual (se enviará por correo).
* Respuesta del sitio: status HTTP, accuracy, aciertos, total, envíos restantes.

**Video de finalista (≤ 8 min)**

1. **Modelado:** algoritmos usados, racional, comparativas.
2. **Evaluación:** métricas, validación, limitaciones.
3. **Aplicación:** caso de uso e impacto.
4. **Diferenciadores:** técnicas, visualizaciones, pipelines, aportes propios.

**Evaluación de la presentación (Etapa 2, 100 pts)**

* **Desarrollo y pruebas (70):**

  * Modelado y validación correctos (20)
  * Creatividad y solidez del modelo (20)
  * Resultados en dataset de prueba (30)
* **Claridad CRISP-DM (10)**
* **Comunicación de resultados (10)**
* **Viabilidad de propuestas y conclusiones (10)**

**Herramientas**

* Sin restricción de lenguaje o librerías.
* Se proveerán scripts ejemplo en **Python**, **R** y **Shell** para envíos.
* Prohibido buscar “ground truth” externo.

**Logística del Congreso**

* Boletos: **preventa \$350**, **venta \$400**, **acceso parcial \$200** (4 pláticas + 1 taller).
* Preventa limitada a **90**.
* Constancias para todos; **premio sorpresa** al equipo ganador.
* Instagram: **congreso.data-science**.
* Se enviarán **dataset**, **link del sitio** y **grabación** de la sesión por correo.

**Checklist inmediato para tu equipo**

1. Leer el **README** y mapear tablas → definir llaves y joins.
2. Separar **train/validación** interno; reservar **test** oficial solo para envíos.
3. Establecer línea base: **estratificación por región** + modelo simple (e.g., árbol/logit).
4. Diseñar **pipeline** reproducible: limpieza, imputación, codificación, normalización.
5. Seleccionar métricas primarias (accuracy obligatoria) + secundarias por clase si procede.
6. Planear **2 envíos/día** como checkpoints, no como exploración ciega.
7. Comenzar guion del **video** con las 4 secciones y evidencias.

**Contactos y soporte**

* Dudas por correo de la mesa; habrá **FAQ** en el sitio.

Listo.
