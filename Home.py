import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
)

# Add title named Enhancing Saptarini, Suasnawa, and Ciptayani's Modified Distributed Genetic Algorithm for Optimized Faculty Workload and Course Assignment
st.title("Enhancing Saptarini, Suasnawa, and Ciptayani's Modified Distributed Genetic Algorithm for Optimized Faculty Workload and Course Assignment")

# Add description that this is made by Joshua D. Bumanlag & Adrian Angelo D. Abelarde
st.subheader("Made by Joshua D. Bumanlag & Adrian Angelo D. Abelarde")

st.write("This is a Streamlit app that showcases our thesis application for the optimization of faculty workload and course assignment. This app shows the difference between the 'Modified DGA' and the 'Proposed DGA'.")

# Divider
st.markdown('---')

# Create columns for people
col1, col2 = st.columns(2)

with col1:
    st.subheader('Joshua D. Bumanlag')
    st.write('Lead and CEO @ GDSC PLM | Notion Campus Leader | IT Engineering Intern @ ING')
    st.image('assets/joshua.jpeg', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    # Add a LinkedIn URL
    st.write("[LinkedIn](https://www.linkedin.com/in/joshuabumanlag/)")

with col2:
    st.subheader('Adrian Angelo D. Abelarde')
    st.write('Web Dev Lead @ GDSC PLM | Freelance Full-stack Developer | Web Dev Intern @ Focus Global INC')
    st.image('assets/adrian.jpg', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    # Add a LinkedIn URL
    st.write("[LinkedIn](https://www.linkedin.com/in/drianlarde/)")