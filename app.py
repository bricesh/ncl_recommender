import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import numpy as np

st.set_option('deprecation.showPyplotGlobalUse', False)

# CSS to inject contained in a string
hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            tbody {align: left}        
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)

@st.cache
def load_class_embeddings():
    with open("class_embeddings", "rb") as class_embeddings_file:
        return np.load(class_embeddings_file)

@st.cache(allow_output_mutation=True)
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

@st.cache(allow_output_mutation=True)
def load_class_data():
    ncl_ext = pd.read_excel('ncl_extended_details.xlsx')
    columns = ['Class','Subclass','Desc']
    ncl_ext.columns = columns

    tree = ET.parse('ncl_en.xml')
    root = tree.getroot()
    class_texts = root.findall('ClassesTexts/ClassTexts')

    ncl_classes = []
    for text_ in class_texts:
        components = []
        components.append([text_.attrib['ncl']])
        components.append([elem_.text for elem_ in text_.findall('Heading/HeadingItem')] + [elem_.text for elem_ in text_.findall('ExplanatoryNote/Introduction')] + [elem_.text for elem_ in text_.findall('ExplanatoryNote/IncludesInParticular/Include')])
        components.append([elem_.text for elem_ in text_.findall('ExplanatoryNote/ExcludesInParticular/Exclude')])
        ncl_classes.append(components)
    
    ncl_inc = []
    for elem in ncl_classes:
        ncl_inc.append(list(map(list, zip(elem[0]*len(elem[1]), elem[1]))))

    ncl_inc = [elem for class_ in ncl_inc for elem in class_]
    ncl_inc = pd.DataFrame(ncl_inc, columns=['Class','Desc'])

    ncl_exc = []
    for elem in ncl_classes:
        ncl_exc.append(list(map(list, zip(elem[0]*len(elem[2]), elem[2]))))

    ncl_exc = [elem for class_ in ncl_exc for elem in class_]
    ncl_exc = pd.DataFrame(ncl_exc, columns=['Class','Desc'])
    
    ncl_inc['Type'] = 'include'
    ncl_exc['Type'] = 'exclude'
    ncl_ext['Type'] = 'specific'

    return pd.concat([ncl_ext[['Class','Desc','Type']], ncl_inc, ncl_exc]).reset_index(drop=True)

def ncl_plot(values_series, labels_series, colour_series):
    plt.style.use('seaborn-v0_8-pastel')
    #print(plt.style.available)
    #plt.style.library['seaborn-v0_8-pastel']['axes.prop_cycle'])
    #print(['#92C6FF']*sum(colour_series=='G') + ['#97F0AA']*sum(colour_series=='S'))

    # set figure size
    plt.figure(figsize=(20,10))

    # plot polar axis
    ax = plt.subplot(111, polar=True)

    # remove grid
    plt.axis('off')

    # Set the coordinates limits
    upperLimit = 1
    lowerLimit = 0.01

    # Compute max and min in the dataset
    max = values_series.max()

    slope = (max - lowerLimit) / max
    heights = slope * values_series + lowerLimit
    width = 2*np.pi / len(values_series.index)

    indexes = list(range(1, len(values_series.index)+1))
    angles = [element * width for element in indexes]
    # Draw bars
    bars = ax.bar(
        x=angles, 
        height=heights, 
        width=width, 
        bottom=lowerLimit,
        linewidth=2,
        color=['#92C6FF']*sum(colour_series=='G') + ['#97F0AA']*sum(colour_series=='S'),
        edgecolor="white")

    # little space between the bar and the label
    labelPadding = 0.001

    # Add labels
    for bar, angle, height, label in zip(bars,angles, heights, labels_series):

        # Labels are rotated. Rotation must be specified in degrees :(
        rotation = np.rad2deg(angle)

        # Flip some labels upside down
        alignment = ""
        if angle >= np.pi/2 and angle < 3*np.pi/2:
            alignment = "right"
            rotation = rotation + 180
        else: 
            alignment = "left"

        # Finally add the labels
        ax.text(
            x=angle, 
            y=lowerLimit + bar.get_height() + labelPadding, 
            s=label, 
            ha=alignment, 
            va='center', 
            rotation=rotation, 
            rotation_mode="anchor")

st.title('Nice Classification Recommender')

nice_class_embeddings = load_class_embeddings()
sbert_model = load_model()
ncl_all = load_class_data()

query = st.sidebar.text_area("Product or Service description", value="A handheld device that allows the user to make phone calls and access applications", max_chars=256, )

with st.sidebar.expander("Examples"):
    st.write('-Protection, safety, and private security agency. We specialize in the areas of close protection, property and home security, and event security')
    st.write('-A handbag is a medium-to-large bag typically used by women to hold personal items. It is often fashionably designed. Versions of the term are purse, pocketbook, pouch')
    st.write('-A Home appliance is any consumer-electronic machine use to complete some household task, such as cooking or cleaning. Home appliances can be classified into: Major appliances (or white goods) and Small appliances')

if query != "":
    query_vec = sbert_model.encode([query])[0]

    ncl_sim = []
    for ncl in nice_class_embeddings:
        ncl_sim.append(cosine(query_vec, ncl))

    ncl_all['similarity'] = ncl_sim
    ncl_all = ncl_all.astype({"Class": int})


    ncl_dist = pd.merge(ncl_all[~(ncl_all['Type'] == 'exclude')].groupby('Class').agg({'similarity': 'max'}).reset_index().sort_values(by=['Class']),
                    ncl_all[ncl_all['Type'] == 'exclude'].groupby('Class').agg({'similarity': 'max'}).reset_index().sort_values(by=['Class']),
                    on='Class')

    ncl_dist['GorS'] = ncl_dist['Class'].apply(lambda x: 'G' if x < 35 else 'S')
    ncl_dist['IncludeLabel'] = ncl_dist.apply(lambda x: '{0:2d}: {1:.2f}'.format(x['Class'], x['similarity_x']), axis=1)
    ncl_dist['ExcludeLabel'] = ncl_dist.apply(lambda x: '{0:2d}: {1:.2f}'.format(x['Class'], x['similarity_y']), axis=1)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Top 3 Goods Classes")
        #st.table(ncl_dist[ncl_dist['GorS'] == 'G'].sort_values(by=['similarity_x'], ascending=False).rename(columns={'Class':'Top 3 Goods Classes'}).head(3)['Top 3 Goods Classes'])
        for _class in ncl_dist[ncl_dist['GorS'] == 'G'].sort_values(by=['similarity_x'], ascending=False).head(3)['Class']:
            st.write(str(_class) + "    [↗](https://www.wipo.int/classifications/nice/nclpub/en/fr/?basic_numbers=show&class_number=" + str(_class) + "&explanatory_notes=show&lang=en&menulang=en&mode=flat&notion=&pagination=no)")

    with col2:
        st.write("Top 3 Services Classes")
        #st.table(ncl_dist[ncl_dist['GorS'] == 'S'].sort_values(by=['similarity_x'], ascending=False).rename(columns={'Class':'Top 3 Services Classes'}).head(3)['Top 3 Services Classes'])
        for _class in ncl_dist[ncl_dist['GorS'] == 'S'].sort_values(by=['similarity_x'], ascending=False).head(3)['Class']:
            st.write(str(_class) + "    [↗](https://www.wipo.int/classifications/nice/nclpub/en/fr/?basic_numbers=show&class_number=" + str(_class) + "&explanatory_notes=show&lang=en&menulang=en&mode=flat&notion=&pagination=no)")

    with st.expander("Class: Similarity"):
        st.write("The graph shows the similarity between the entered description and the description of the 45 Nice classes. The similarity is a number between 0 and 1, with 1 representing high similarity.")
        st.pyplot(ncl_plot(ncl_dist['similarity_x'], ncl_dist['IncludeLabel'], ncl_dist['GorS']))

    with st.expander("Top 15 'Include in particular' matches"):
        st.write("The table lists the top 15 matches to the 'Include in particular' descriptions for each class.")
        st.table(ncl_all[(ncl_all['Type'] == 'include')].sort_values(by=['similarity'], ascending=False)[['Class','Desc','similarity']].head(15).style.set_properties(subset=['Desc'], **{'width-min': '50px'}))
    
    with st.expander("Top 15 'Indication' matches"):
        st.write("The table lists the top 15 matches to the 'Indications' for each class.")
        st.table(ncl_all[(ncl_all['Type'] == 'specific')].sort_values(by=['similarity'], ascending=False)[['Class','Desc','similarity']].head(15).style.set_properties(subset=['Desc'], **{'width-min': '50px'}))

    with st.expander("Top 15 'Does not include in particular' matches"):
        st.write("The table lists the top 15 matches to the 'Does not include in particular' descriptions for each class.")
        st.table(ncl_all[(ncl_all['Type'] == 'exclude')].sort_values(by=['similarity'], ascending=False)[['Class','Desc','similarity']].head(15).style.set_properties(subset=['Desc'], **{'width-min': '50px'}))
else:
    st.write("Enter a product or service description")
    