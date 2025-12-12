import streamlit as st
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.query_enhance.query_intelligence import enhance_query
from src.searching.search_engine import search

st.title("Medical Information Retrieval System")

# Input query from user
query = st.text_input("Query here :")

# Enhanced query display
enhanced_query = ""
if query.strip():
    enhanced_query = enhance_query(query)
    st.text_area("Enhanced Query :", enhanced_query, height=120)

# Search button and display results
if st.button("Search Information"):
    if not query.strip():
        st.warning("Please enter a query first!")
    elif not enhanced_query:
        st.error("Enhanced query missing.")
    else:
        results = search(enhanced_query)

        st.write("---")
        st.write("###  Search Results:")
       
        for r in results:
            st.write(f"**Score:** {r['score']:.4f}")
            st.write(r["text"])
            st.write("---")