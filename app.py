import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
from scipy import integrate
import warnings
warnings.filterwarnings('ignore')

def evaluar_funcion(func_str, x_val, y_val):
    """
    Evalúa una función string en puntos específicos
    """
    try:
        # Reemplazar variables por valores
        func_str = func_str.replace('x', f'({x_val})')
        func_str = func_str.replace('y', f'({y_val})')
        func_str = func_str.replace('sin', 'np.sin')
        func_str = func_str.replace('cos', 'np.cos')
        func_str = func_str.replace('tan', 'np.tan')
        func_str = func_str.replace('exp', 'np.exp')
        func_str = func_str.replace('log', 'np.log')
        func_str = func_str.replace('sqrt', 'np.sqrt')
        func_str = func_str.replace('^', '**')
        
        return eval(func_str)
    except:
        return 0

def crear_funcion_numerica(func_str):
    """
    Crea una función numérica a partir de un string
    """
    def f(x, y):
        try:
            # Manejar arrays
            if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
                x = np.atleast_1d(x)
                y = np.atleast_1d(y)
                result = np.zeros_like(x, dtype=float)
                
                for i, (xi, yi) in enumerate(zip(x.flat, y.flat)):
                    result.flat[i] = evaluar_funcion(func_str, xi, yi)
                return result
            else:
                return evaluar_funcion(func_str, x, y)
        except:
            return 0
    return f

def calcular_integral_doble_numerica(func_str, x_min, x_max, y_min, y_max):
    """
    Calcula la integral doble usando métodos numéricos
    """
    try:
        f = crear_funcion_numerica(func_str)
        resultado, error = integrate.dblquad(f, x_min, x_max, lambda x: y_min, lambda x: y_max)
        return resultado, error
    except Exception as e:
        return None, str(e)

def crear_grafico_3d(func_str, x_min, x_max, y_min, y_max, resultado=None):
    """
    Crea un gráfico 3D de la función
    """
    try:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Crear malla de puntos
        x = np.linspace(x_min, x_max, 50)
        y = np.linspace(y_min, y_max, 50)
        X, Y = np.meshgrid(x, y)
        
        # Evaluar la función
        f = crear_funcion_numerica(func_str)
        Z = f(X, Y)
        
        # Crear superficie
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                              linewidth=0, antialiased=True)
        
        # Proyección en el plano XY
        ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), cmap='viridis', alpha=0.5)
        
        # Configuración del gráfico
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('f(x,y)', fontsize=12)
        ax.set_title(f'Función: f(x,y) = {func_str}\n' + 
                    (f'Integral = {resultado:.6f}' if resultado is not None else ''),
                    fontsize=14)
        
        # Barra de colores
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        return fig
    except Exception as e:
        st.error(f"Error al crear el gráfico: {str(e)}")
        return None

def crear_grafico_contorno(func_str, x_min, x_max, y_min, y_max, resultado=None):
    """
    Crea un gráfico de contorno 2D
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Crear malla de puntos
        x = np.linspace(x_min, x_max, 100)
        y = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x, y)
        
        # Evaluar la función
        f = crear_funcion_numerica(func_str)
        Z = f(X, Y)
        
        # Crear contorno
        contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
        contour_lines = ax.contour(X, Y, Z, levels=20, colors='black', alpha=0.3, linewidths=0.5)
        
        # Marcar la región de integración
        ax.add_patch(plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, 
                                  fill=False, edgecolor='red', linewidth=3, linestyle='--'))
        
        # Configuración del gráfico
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_title(f'Contorno de f(x,y) = {func_str}\n' + 
                    (f'Integral = {resultado:.6f}' if resultado is not None else ''),
                    fontsize=14)
        
        # Barra de colores
        plt.colorbar(contour)
        
        # Leyenda para la región
        ax.text(0.02, 0.98, f'Región: [{x_min}, {x_max}] × [{y_min}, {y_max}]', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return fig
    except Exception as e:
        st.error(f"Error al crear el gráfico de contorno: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Calculadora de Integrales Dobles", page_icon="∬", layout="wide")
    
    st.title("∬ Calculadora de Integrales Dobles")
    st.markdown("---")
    
    # Información teórica
    with st.expander("ℹ️ Información sobre Integrales Dobles"):
        st.markdown("""
        **Una integral doble** calcula el volumen bajo una superficie f(x,y) sobre una región R:
        
        **∬[R] f(x,y) dA = ∫[a,b] ∫[c,d] f(x,y) dy dx**
        
        **Interpretación geométrica:**
        - Si f(x,y) ≥ 0, la integral representa el volumen bajo la superficie
        - Si f(x,y) < 0, representa el volumen "negativo"
        - El resultado neto es la diferencia entre volúmenes positivos y negativos
        
        **Sintaxis de funciones:**
        - Operaciones: +, -, *, /, ** (potencia)
        - Funciones: sin, cos, tan, exp, log, sqrt
        - Constantes: pi, e
        - Ejemplos: x**2 + y**2, sin(x*y), exp(x+y)
        """)
    
    # Layout en columnas
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Definición de la Integral")
        
        # Función
        func_str = st.text_input("Función f(x,y):", value="x**2 + y**2", 
                                help="Ingresa la función en términos de x e y")
        
        # Límites de integración
        st.write("**Límites de integración:**")
        col1a, col1b = st.columns(2)
        with col1a:
            x_min = st.number_input("x mínimo:", value=-1.0, step=0.1)
            x_max = st.number_input("x máximo:", value=1.0, step=0.1)
        with col1b:
            y_min = st.number_input("y mínimo:", value=-1.0, step=0.1)
            y_max = st.number_input("y máximo:", value=1.0, step=0.1)
        
        # Validación
        if x_min >= x_max:
            st.error("El límite inferior de x debe ser menor que el superior")
        elif y_min >= y_max:
            st.error("El límite inferior de y debe ser menor que el superior")
        else:
            # Botón para calcular
            if st.button("🔢 Calcular Integral", type="primary"):
                with st.spinner("Calculando integral..."):
                    resultado, error = calcular_integral_doble_numerica(func_str, x_min, x_max, y_min, y_max)
                    
                    if resultado is not None:
                        st.success(f"**Resultado: {resultado:.8f}**")
                        st.info(f"Error estimado: ±{error:.2e}")
                        
                        # Interpretación
                        if abs(resultado) < 1e-10:
                            st.info("🔄 El resultado es aproximadamente cero")
                        elif resultado > 0:
                            st.info("📈 Volumen neto positivo")
                        else:
                            st.info("📉 Volumen neto negativo")
                        
                        # Información adicional
                        area_region = (x_max - x_min) * (y_max - y_min)
                        valor_promedio = resultado / area_region
                        st.write(f"**Área de la región:** {area_region:.4f}")
                        st.write(f"**Valor promedio de f:** {valor_promedio:.6f}")
                        
                        # Guardar en session state para usar en gráficos
                        st.session_state.ultimo_resultado = resultado
                    else:
                        st.error(f"Error en el cálculo: {error}")
        
        # Ejemplos predefinidos
        st.markdown("---")
        st.subheader("📚 Ejemplos Predefinidos")
        
        ejemplos = {
            "Paraboloide": {"func": "x**2 + y**2", "x_min": -1, "x_max": 1, "y_min": -1, "y_max": 1},
            "Función trigonométrica": {"func": "sin(x) * cos(y)", "x_min": 0, "x_max": 3.14159, "y_min": 0, "y_max": 3.14159},
            "Función exponencial": {"func": "exp(-(x**2 + y**2))", "x_min": -2, "x_max": 2, "y_min": -2, "y_max": 2},
            "Función lineal": {"func": "x + y", "x_min": 0, "x_max": 2, "y_min": 0, "y_max": 2},
            "Función constante": {"func": "5", "x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1}
        }
        
        ejemplo_seleccionado = st.selectbox("Selecciona un ejemplo:", list(ejemplos.keys()))
        
        if st.button("🔄 Cargar Ejemplo"):
            ej = ejemplos[ejemplo_seleccionado]
            st.session_state.func_ejemplo = ej["func"]
            st.session_state.x_min_ejemplo = ej["x_min"]
            st.session_state.x_max_ejemplo = ej["x_max"]
            st.session_state.y_min_ejemplo = ej["y_min"]
            st.session_state.y_max_ejemplo = ej["y_max"]
            st.experimental_rerun()
    
    with col2:
        st.subheader("📊 Visualización")
        
        # Opciones de visualización
        tipo_grafico = st.radio("Tipo de gráfico:", ["Superficie 3D", "Contorno 2D", "Ambos"])
        
        if st.button("🎨 Generar Gráfico") or st.checkbox("Actualizar automáticamente", value=True):
            resultado = getattr(st.session_state, 'ultimo_resultado', None)
            
            if tipo_grafico == "Superficie 3D":
                fig = crear_grafico_3d(func_str, x_min, x_max, y_min, y_max, resultado)
                if fig:
                    st.pyplot(fig)
            
            elif tipo_grafico == "Contorno 2D":
                fig = crear_grafico_contorno(func_str, x_min, x_max, y_min, y_max, resultado)
                if fig:
                    st.pyplot(fig)
            
            else:  # Ambos
                fig_3d = crear_grafico_3d(func_str, x_min, x_max, y_min, y_max, resultado)
                if fig_3d:
                    st.pyplot(fig_3d)
                
                fig_2d = crear_grafico_contorno(func_str, x_min, x_max, y_min, y_max, resultado)
                if fig_2d:
                    st.pyplot(fig_2d)
    
    # Información adicional
    st.markdown("---")
    st.markdown("💡 **Consejos:**")
    st.markdown("- Usa ** para potencias (ej: x**2 en lugar de x^2)")
    st.markdown("- Las funciones trigonométricas están en radianes")
    st.markdown("- Para funciones complejas, verifica que la sintaxis sea correcta")
    st.markdown("- El cálculo puede tardar más para funciones complicadas o regiones grandes")

if __name__ == "__main__":
    main()