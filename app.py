import streamlit as st
from data import Data
from streamlit_image_select import image_select
import serializer
import requests


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
# Prédiction de masque
Choisissez une image dans la liste ci dessous et comparez le masque réelle au masque prédit par le model
""")

if "data" not in st.session_state:
    st.session_state['data'] = Data()

if st.button('Afficher 4 images au hazard'):
    st.session_state['data'].shuffle()

img_list = list(st.session_state['data'].df.image[0:4])
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

    response = requests.post(
        "https://api-predict.codeheures.fr/",
        headers={
            'Content-type': 'application/json',
            'Accept': 'application/json'
        },
        json={"image": json})
    json_response = response.json()
    serialized_predited_mask = json_response["predicted_mask"]
    predicted_mask = serializer.deserialize(serialized_predited_mask)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('## Image')
        st.image(image)
    with col2:
        st.markdown('## Masque réel')
        st.image(mask)
    with col3:
        st.markdown('## Masque prédit')
        st.image(predicted_mask)
