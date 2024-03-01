import streamlit as st
from data import Data
from streamlit_image_select import image_select
import serializer
import api

st.set_page_config(
    page_title="Prediction d'un masque",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)


@st.cache_data
def get_dt():
    return Data()


st.write("""
# My first app
Hello *world!*
""")

if "data" not in st.session_state:
    st.session_state['data'] = Data()

if st.button('Afficher 16 images au hazard'):
    st.session_state['data'].shuffle()

img_list = list(st.session_state['data'].df.image[0:16])
images = [st.session_state['data'].images_dir + name for name in img_list]

idx = image_select(
    label="Choisir une",
    images=images,
    use_container_width=False,
    return_value="index",
    index=-1
)

if idx >= 0:
    image = st.session_state['data'].get_image(idx)
    mask = st.session_state['data'].get_mask(idx, mask_to_y=False)
    json = serializer.serialize(image)

    predited_mask = api.predict(json)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('## Image')
        st.image(image)
    with col2:
        st.markdown('## Masque réelle')
        st.image(mask)
    with col3:
        st.markdown('## Masque prédit')
        st.image(predited_mask)
