# Configuración de Google Generative AI (Gemini)

## Obtener API Key Gratis

### Paso 1: Ir a Google AI Studio
1. Accede a https://aistudio.google.com
2. Inicia sesión con tu cuenta de Google (crea una si es necesario)

### Paso 2: Crear API Key
1. En la página principal, haz clic en **"Get API key"** o **"Create API key"**
2. Selecciona **"Create API key in new project"**
3. Google generará una clave automáticamente

### Paso 3: Copiar la API Key
1. Copia la clave mostrada
2. No la compartas públicamente

## Configurar la Variable de Entorno

### Opción A: Variable de Entorno Local (Linux/Mac)
```bash
export GOOGLE_API_KEY="tu_api_key_aqui"
```

### Opción B: Variable de Entorno Local (Windows)
En PowerShell:
```powershell
$env:GOOGLE_API_KEY="tu_api_key_aqui"
```

En Command Prompt:
```cmd
set GOOGLE_API_KEY=tu_api_key_aqui
```

### Opción C: Archivo .env
Crea un archivo `.env` en la raíz del proyecto:
```
GOOGLE_API_KEY=tu_api_key_aqui
```

Y carga con:
```python
from dotenv import load_dotenv
load_dotenv()
```

## Instalación de Dependencias

```bash
pip install google-generativeai
```

## Límites de Uso

El modelo gratuito **Gemini 1.5 Flash** tiene límites:
- **15 requests por minuto** (inicio)
- **1 millón de tokens por día**

Para más información: https://ai.google.dev/pricing

## Verificación

Para verificar que está configurado correctamente:
```python
from app.ai.gemini_recommendations import generate_city_recommendations

result = generate_city_recommendations("París")
print(result)
```

Si ves recomendaciones, ¡está funcionando!
