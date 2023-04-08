from collections import OrderedDict

import streamlit as st

# TODO : change TITLE, TEAM_MEMBERS and PROMOTION values in config.py.
import config
from tabs import DATASET.py, DataViz.py, RFM.py, Kmeans.py, Prediction.py


st.set_page_config(
    page_title=config.TITLE,
    page_icon="https://datascientest.com/wp-content/uploads/2020/03/cropped-favicon-datascientest-1-32x32.png",
)

with open("style.css", "r") as f:
    style = f.read()

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)


# TODO: add new and/or renamed tab in this ordered dict by
# passing the name in the sidebar as key and the imported tab
# as value as follow :
TABS = OrderedDict(
    [
        (DATASET.sidebar_name, DATASET),
        (DataViz.sidebar_name, DataViz),
        (RFM.sidebar_name, RFM),
        (Kmeans.sidebar_name, Kmeans),
        (Prediction.sidebar_name, Prediction),
        
    ]
)


def run():
    st.sidebar.image(
        "https://studio.datascientest.com/static/images/logo_header.svg",
        width=300,
    )
    tab_name = st.sidebar.radio("", list(TABS.keys()), 0)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {config.PROMOTION}")

    st.sidebar.markdown("### Team members:")
    for member in config.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)

    tab = TABS[tab_name]

    tab.run()

 
if __name__ == "__main__":
    run()
