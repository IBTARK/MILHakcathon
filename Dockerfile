# ─────────────────────────────────────────────
# 1. Imagen base: Python oficial, ligera
# ─────────────────────────────────────────────
FROM python:3.12-slim  

# ─────────────────────────────────────────────
# 2. Crea un usuario sin privilegios
#    (recomendación de HF para evitar correr como root)
# ─────────────────────────────────────────────
RUN useradd -m -u 1000 user
USER user

# ─────────────────────────────────────────────
# 3. Variables y directorio de trabajo
# ─────────────────────────────────────────────
ENV HOME=/home/user \
    PATH=$HOME/.local/bin:$PATH
WORKDIR $HOME/app            

# ─────────────────────────────────────────────
# 4. Instala dependencias Python
# ─────────────────────────────────────────────
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# ─────────────────────────────────────────────
# 5. Copia el resto del código del proyecto
# ─────────────────────────────────────────────
COPY --chown=user . .

# ─────────────────────────────────────────────
# 6. Comando de arranque
#    -h   → evita que Chainlit intente abrir el navegador
#    --host 0.0.0.0 → expone fuera del contenedor (obligatorio en Spaces)
#    --port 7860    → debe coincidir con app_port del README
# ─────────────────────────────────────────────
CMD ["chainlit", "run", "app.py", "-h", "--host", "0.0.0.0", "--port", "7860"]
