#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 14:21:05 2021

@author: baptiste
"""

import streamlit as st
st.set_page_config(page_title='Détecteur de COVID v1')

# Local Imports
import intro
import biais
import preprocessing
import application

# Sidebar Options & File Uplaod
las_file=None
st.sidebar.write('# Détecteur de COVID')


# Sidebar Navigation
st.sidebar.title('Menu')
options = st.sidebar.radio('Selectionnez une page :', 
    ['Introduction', 'Biais du dataset', 'Preprocessing', 'Application'])

if options == 'Introduction':
    intro.intro()
elif options == 'Biais du dataset':
    biais.biais()
elif options == 'Preprocessing':
    preprocessing.preprocessing()
elif options == 'Application':
    application.application()