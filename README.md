# Proyecto integrador de Ciencia de Datos

Repositorio git propio, anidado en el repositorio de la maestría. Las filas individuales del dataset no se versionan: el `.gitignore` de esta carpeta ignora los datos crudos y los entornos virtuales, y solo deja pasar los agregados de `data/agregados/`.

Estructura, según la sección 3.4 de la guía:

- `data/agregados/`: solo datos agregados, que el dashboard lee desde el repositorio. Nunca filas individuales del dataset.
- `data/`: el resto de esta carpeta no se versiona. El dataset original vive en `../dataset/saber11/`.
- `notebooks/`: cuadernos de las fases 2 y 3.
- `src/`: código reutilizable de preprocesamiento, modelos y MLP.
- `dashboard/`: aplicación de Streamlit, Dash, Panel o Shiny, o el archivo de Power BI o Tableau Public.
- `requirements.txt`: versiones exactas al cerrar el proyecto.
