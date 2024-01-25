import streamlit as st


def main():
    st.title('Ejemplo con Streamlit')
    st.write('Selecciona tus preferencias:')

    color = st.selectbox(
        'Elige tu color favorito',
        ['Rojo', 'Azul', 'Verde']
    )
    st.write(f'Has elegido el color: {color}')

    numero = st.slider('Selecciona un número', 1, 10)
    st.write('Número seleccionado:', numero)

    if st.button('Haz click aquí'):
        st.write('¡Botón presionado!')


if __name__ == '__main__':
    main()
